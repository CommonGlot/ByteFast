// Native, dependency-free ByteFast prediction and training bindings.
// Original fastText C++ implementation is preserved in its own legacy module.
// Copyright (c) 2026 CommonGlot contributors. MIT License.
#include "bytefast_calibrated.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

namespace py = pybind11;

using Predictions = std::pair<std::vector<std::string>,std::vector<float>>;

Predictions pickTop(const std::vector<std::string>& labels,
                    const std::vector<float>& probs,size_t k,float threshold) {
  std::vector<uint32_t> indices(probs.size());
  std::iota(indices.begin(),indices.end(),0);
  const size_t n=std::min(k,indices.size());
  std::partial_sort(indices.begin(),indices.begin()+n,indices.end(),
        [&](uint32_t a,uint32_t b){
          return probs[a]==probs[b]?a<b:probs[a]>probs[b];
        });
  Predictions result;
  for(size_t i=0;i<n;++i) {
    const auto id=indices[i];
    if(probs[id]<threshold)continue;
    result.first.push_back("__label__"+labels[id]);
    result.second.push_back(probs[id]);
  }
  return result;
}

class ByteModelBinding {
 public:
  Model model;
  void load_model(const std::string& path){model.load(path);}
  void save_model(const std::string& path)const{model.save(path);}
  std::vector<std::string> get_labels()const{
    auto labels=model.labels;
    for(auto& l:labels)l="__label__"+l;
    return labels;
  }
  int get_dimension()const{return static_cast<int>(model.opt.dim);}
  Predictions predict(const std::string& text,size_t k=1,float threshold=0)const{
    if(!model.labels.size())throw std::runtime_error("Model not trained or loaded");
    return pickTop(model.labels,
       model.probabilities(model.embedding(model.features(text))),k,threshold);
  }
  void train(const std::string& path, uint32_t dim,uint32_t bucket,
             uint32_t minn,uint32_t maxn,uint32_t epochs,float lr,
             bool ascii_fold,uint32_t max_bytes) {
    if(dim<1||dim>4096||bucket<1||bucket>10000000||minn<1||maxn>16||
       minn>maxn||epochs<1||epochs>1000||max_bytes<1||
       !std::isfinite(lr)||lr<=0||lr>10)
       throw std::runtime_error("Invalid byte training hyperparameters");
    model.opt.dim=dim;model.opt.bucket=bucket;
    model.opt.minN=minn;model.opt.maxN=maxn;
    model.opt.epochs=epochs;model.opt.lr=lr;
    model.opt.maxBytes=max_bytes;model.opt.asciiFold=ascii_fold?1:0;
    const auto examples=readRecords(path);
    model.train(examples);
  }
};

class CalibratedBinding {
 public:
  Model base;
  Lexicon lex;
  Fusion fused;
  void load(const std::string& basepath,
            const std::string& lexicalpath,
            const std::string& fusionpath) {
    base.load(basepath);
    lex.load(base,lexicalpath);
    fused.load(base,lex,fusionpath);
  }
  std::vector<std::string> get_labels()const{
    std::vector<std::string> out;
    out.reserve(fused.labels.size());
    for(const auto& l:fused.labels)out.push_back("__label__"+l);
    return out;
  }
  int get_dimension()const{return static_cast<int>(base.opt.dim);}
  Predictions predict(const std::string& text,size_t k=1,
                      float threshold=0,float gate=0.75f)const {
    if(fused.labels.empty())throw std::runtime_error("Calibrated model not loaded");
    if(!std::isfinite(gate)||gate<0||gate>2)
      throw std::runtime_error("Invalid ambiguity gate");
    const auto baseP=base.probabilities(base.embedding(base.features(text)));
    uint32_t first=0;float high=-1,second=-1;
    for(uint32_t i=0;i<baseP.size();++i){
      if(baseP[i]>high){second=high;high=baseP[i];first=i;}
      else if(baseP[i]>second)second=baseP[i];
    }
    if(high-second>=gate)
      return pickTop(fused.labels,baseP,k,threshold);
    const auto words=lex.evidence(text);
    std::vector<float> p(baseP.size());
    float largest=-std::numeric_limits<float>::infinity();
    for(size_t i=0;i<p.size();++i) {
      const float s=static_cast<float>(
         fused.theta[0]*std::log(std::max(baseP[i],1e-20f))+
         (fused.theta[1]+fused.theta[2+i])*words[i]+
         fused.theta[2+baseP.size()+i]);
      p[i]=s;largest=std::max(largest,s);
    }
    float sum=0;
    for(float& v:p){v=std::exp(v-largest);sum+=v;}
    for(float& v:p)v/=sum;
    return pickTop(fused.labels,p,k,threshold);
  }
};

// Train lexical evidence and its supervised fusion parameters after the
// base byte model has been fitted on a disjoint training partition.
void fitLexicalSidecars(
    const std::string& baseFile, const std::string& trainFile,
    const std::string& calibrationFile, const std::string& validationFile,
    const std::string& lexiconOutput, const std::string& fusionOutput,
    uint32_t vocabLimit, uint32_t minDocumentFrequency,
    unsigned fusionEpochs, double fusionStep, double fusionL2,
    unsigned fusionPatience) {
  if(vocabLimit<1||vocabLimit>1000000||minDocumentFrequency<1)
    throw std::invalid_argument("Invalid lexical vocabulary settings");
  Model base;
  base.load(baseFile);
  Lexicon lexicon;
  lexicon.train(base,readRecords(trainFile),vocabLimit,minDocumentFrequency);
  if(lexicon.entries.empty())
    throw std::runtime_error("Lexical selection produced an empty vocabulary");
  lexicon.save(lexiconOutput);
  FusionOptions options;
  options.maxEpochs=fusionEpochs;options.step=fusionStep;
  options.l2=fusionL2;options.patience=fusionPatience;
  Fusion fusion;
  fusion.fit(base,lexicon,readRecords(calibrationFile),
             readRecords(validationFile),options);
  fusion.save(fusionOutput);
}

PYBIND11_MODULE(_proto_pybind,m) {
  m.doc()="ByteFast UTF-8 byte-based text classifiers with optional learned lexical fusion";
  m.def("fit_lexical_sidecars",&fitLexicalSidecars,
        py::arg("base_model"),py::arg("train_data"),
        py::arg("calibration_data"),py::arg("validation_data"),
        py::arg("lexicon_output"),py::arg("fusion_output"),
        py::arg("lexical_vocab"),py::arg("lexical_min_df"),
        py::arg("fusion_epochs")=90,py::arg("fusion_lr")=0.07,
        py::arg("fusion_l2")=0.01,py::arg("fusion_patience")=16,
        py::call_guard<py::gil_scoped_release>());
  py::class_<ByteModelBinding>(m,"ByteFastModel")
    .def(py::init<>())
    .def("load_model",&ByteModelBinding::load_model)
    .def("save_model",&ByteModelBinding::save_model)
    .def("predict",&ByteModelBinding::predict,
         py::arg("text"),py::arg("k")=1,py::arg("threshold")=0.f)
    .def("get_labels",&ByteModelBinding::get_labels)
    .def("get_dimension",&ByteModelBinding::get_dimension)
    .def("train",&ByteModelBinding::train,
         py::arg("input"),py::arg("dim")=64,py::arg("bucket")=65536,
         py::arg("minn")=2,py::arg("maxn")=5,
         py::arg("epoch")=5,py::arg("lr")=.15f,
         py::arg("ascii_fold")=false,py::arg("max_bytes")=512);
  py::class_<CalibratedBinding>(m,"CalibratedByteFastModel")
    .def(py::init<>())
    .def("load_model",&CalibratedBinding::load)
    .def("predict",&CalibratedBinding::predict,
         py::arg("text"),py::arg("k")=1,py::arg("threshold")=0.f,
         py::arg("gate")=.75f)
    .def("get_labels",&CalibratedBinding::get_labels)
    .def("get_dimension",&CalibratedBinding::get_dimension);
}
