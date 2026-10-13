// Supervised byte/lexical fusion for CPU language identification.
// Compatible with the existing ByteFast byte model and sparse word sidecar.
// Fits a shared byte temperature, shared word scale, per-language lexical
// offsets and per-language biases using train-only calibration examples.
#include "bytefast_lexical.h"
#include <array>
#include <sstream>

namespace {
struct Example {
  std::vector<float> b, z;
  uint32_t label;
};
struct Fusion {
  std::vector<std::string> labels;
  std::vector<double> params;
  size_t count() const {return labels.size();}
  void init(const std::vector<std::string>& langs) {
    labels=langs;
    params.assign(2+2*count(),0.);
    params[0]=1.; // byte log-probability coefficient
    params[1]=1.; // learned shared lexical coefficient, not a fixed multiplier
  }
  void check(const Model& m,const Lexicon& lex) const {
    if(labels!=m.labels || lex.labels!=m.labels || params.size()!=2+2*count())
      throw std::runtime_error("Fusion, lexical and byte models must share identical labels");
  }
  std::vector<double> logits(const Example& e) const {
    std::vector<double> s(count());
    for(size_t i=0;i<count();++i) {
      s[i]=params[0]*e.b[i]+
            (params[1]+params[2+i])*e.z[i]+params[2+count()+i];
    }
    return s;
  }
  static double logLoss(const std::vector<double>& logits,uint32_t label) {
    const double hi=*std::max_element(logits.begin(),logits.end());
    double denom=0.;
    for(double s:logits)denom+=std::exp(s-hi);
    return std::log(denom)+hi-logits[label];
  }
  Example encode(const Model& m,const Lexicon& lex,
                 const std::string& text,uint32_t y=0) const {
    Example e;
    const auto p=m.probabilities(m.embedding(m.features(text)));
    e.b.resize(p.size());
    for(size_t i=0;i<p.size();++i)
      e.b[i]=std::log(std::max(1e-20f,p[i]));
    e.z=lex.evidence(text);
    e.label=y;
    return e;
  }
  uint32_t predict(const Model& m,const Lexicon& lex,
                   const std::string& text,float gate) const {
    const auto p=m.probabilities(m.embedding(m.features(text)));
    uint32_t top=0;
    float first=-1.f,second=-1.f;
    for(size_t c=0;c<p.size();++c) {
      if(p[c]>first){second=first;first=p[c];top=static_cast<uint32_t>(c);}
      else if(p[c]>second)second=p[c];
    }
    if(first-second>=gate)return top;
    Example e;
    e.b.resize(p.size());
    for(size_t i=0;i<p.size();++i)e.b[i]=std::log(std::max(p[i],1e-20f));
    e.z=lex.evidence(text);
    const auto v=logits(e);
    return static_cast<uint32_t>(std::max_element(v.begin(),v.end())-v.begin());
  }
  void train(const Model& m,const Lexicon& lex,
             const std::vector<Record>& records,uint32_t epochs,double learningRate) {
    init(m.labels);
    std::unordered_map<std::string,uint32_t> labelIds;
    for(uint32_t i=0;i<count();++i) labelIds[labels[i]]=i;
    std::vector<Example> trainData,validData;
    std::vector<uint32_t> seen(count());
    for(const Record& r:records) {
      auto it=labelIds.find(r.label);
      if(it==labelIds.end())continue;
      const uint32_t c=it->second;
      auto e=encode(m,lex,r.text,c);
      // Every fifth occurrence per label is validation-only.
      // This split is created from official WiLI TRAIN, never from test.
      if(seen[c]++%5==0) validData.push_back(std::move(e));
      else trainData.push_back(std::move(e));
    }
    if(trainData.empty() || validData.empty())
      throw std::runtime_error("Not enough fusion calibration examples");
    std::cerr<<"FUSION_CALIB train="<<trainData.size()
             <<" validation="<<validData.size()<<" labels="<<count()<<'\n';
    std::vector<double> moment(params.size()),variance(params.size());
    auto bestParams=params;
    double bestLoss=std::numeric_limits<double>::infinity();
    unsigned noImprove=0;
    uint32_t bestEpoch=0;
    for(uint32_t epoch=1;epoch<=epochs;++epoch) {
      std::vector<double> gradient(params.size());
      double trainLoss=0;
      for(const auto& e:trainData) {
        const auto s=logits(e);
        const double highest=*std::max_element(s.begin(),s.end());
        std::vector<double> probs(count());
        double denom=0;
        for(size_t c=0;c<count();++c) {
          probs[c]=std::exp(s[c]-highest);denom+=probs[c];
        }
        trainLoss+=std::log(denom)+highest-s[e.label];
        for(size_t c=0;c<count();++c) {
          const double err=probs[c]/denom-(c==e.label?1.:0.);
          gradient[0]+=err*e.b[c];
          gradient[1]+=err*e.z[c];
          gradient[2+c]+=err*e.z[c];
          gradient[2+count()+c]+=err;
        }
      }
      for(double& g:gradient)g/=trainData.size();
      // Stable shrinkage toward one common lexical weight. Rare labels cannot
      // freely memorize their tiny calibration samples with huge corrections.
      for(size_t c=0;c<count();++c) {
        gradient[2+c]+=0.02*params[2+c]/count();
        gradient[2+count()+c]+=0.01*params[2+count()+c]/count();
      }
      gradient[0]+=0.0001*(params[0]-1.);
      gradient[1]+=0.0001*params[1];

      // Full-batch Adam: ~2+2*C trainable parameters. All language-specific
      // coefficients are fitted by supervised cross-entropy.
      for(size_t i=0;i<params.size();++i) {
        moment[i]=0.9*moment[i]+0.1*gradient[i];
        variance[i]=0.999*variance[i]+0.001*gradient[i]*gradient[i];
        const double mm=moment[i]/(1-std::pow(0.9,epoch));
        const double vv=variance[i]/(1-std::pow(0.999,epoch));
        params[i]-=learningRate*mm/(std::sqrt(vv)+1e-8);
      }
      params[0]=std::clamp(params[0],0.05,5.);
      params[1]=std::clamp(params[1],0.,8.);
      for(size_t c=0;c<count();++c) {
        params[2+c]=std::clamp(params[2+c],-4.,4.);
        params[2+count()+c]=std::clamp(params[2+count()+c],-3.,3.);
      }
      double valLoss=0;
      size_t valCorrect=0;
      for(const auto& e:validData) {
        const auto s=logits(e);
        valLoss+=logLoss(s,e.label);
        valCorrect+=(static_cast<size_t>(
          std::max_element(s.begin(),s.end())-s.begin())==e.label);
      }
      valLoss/=validData.size();
      if(valLoss<bestLoss-1e-5) {
        bestLoss=valLoss;bestParams=params;bestEpoch=epoch;noImprove=0;
      } else ++noImprove;
      if(epoch==1 || epoch%5==0 || noImprove>=10)
        std::cerr<<"FUSION_EPOCH epoch="<<epoch
                 <<" train_ce="<<trainLoss/trainData.size()
                 <<" validation_ce="<<valLoss
                 <<" validation_accuracy="<<double(valCorrect)/validData.size()
                 <<" byte_weight="<<params[0]
                 <<" lexical_global_weight="<<params[1]<<'\n';
      if(noImprove>=10)break;
    }
    params=std::move(bestParams);
    double minWord=std::numeric_limits<double>::infinity();
    double maxWord=-minWord;
    for(size_t c=0;c<count();++c) {
      minWord=std::min(minWord,params[1]+params[2+c]);
      maxWord=std::max(maxWord,params[1]+params[2+c]);
    }
    std::cerr<<"FUSION_FIT best_epoch="<<bestEpoch<<" validation_ce="<<bestLoss
             <<" byte_weight="<<params[0]<<" lexical_global_weight="<<params[1]
             <<" lexical_label_weight_min="<<minWord
             <<" lexical_label_weight_max="<<maxWord<<'\n';
  }
  void save(const std::string& path) const {
    std::ofstream out(path,std::ios::binary);
    if(!out)throw std::runtime_error("Could not write fusion file");
    Model::write(out,uint32_t{0x42465553U});
    Model::write(out,uint32_t{1});
    Model::write(out,static_cast<uint32_t>(count()));
    for(const auto& s:labels) {
      if(s.size()>256)throw std::runtime_error("Invalid label string");
      Model::write(out,static_cast<uint32_t>(s.size()));
      out.write(s.data(),s.size());
    }
    for(double p:params) {
      const float v=static_cast<float>(p);
      Model::write(out,v);
    }
    if(!out)throw std::runtime_error("Fusion write failed");
  }
  void load(const Model& base,const Lexicon& lex,const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("Could not read fusion file");
    uint32_t magic,version,n;
    Model::read(in,magic);Model::read(in,version);Model::read(in,n);
    if(magic!=0x42465553U || version!=1 || n!=base.labels.size())
      throw std::runtime_error("Invalid fusion header or label count");
    labels.clear();
    for(uint32_t i=0;i<n;++i) {
      uint32_t len;Model::read(in,len);
      if(len>256)throw std::runtime_error("Invalid label size in fusion");
      std::string label(len,'\0');
      if(len && !in.read(&label[0],len))throw std::runtime_error("Truncated labels");
      labels.push_back(std::move(label));
    }
    check(base,lex); // verifies label order and lexical vocabulary compatibility
    params.resize(2+2*n);
    for(double& p:params) {
      float v;Model::read(in,v);
      if(!std::isfinite(v)||std::abs(v)>100)
        throw std::runtime_error("Invalid fusion coefficient");
      p=v;
    }
  }
};
void usageFusion() {
  std::cerr<<"Usage:\n"
   <<"  bytefast-fusion fit BASE.bin WORDS.lex CALIBRATION.fasttext OUT.fuse [-epochs 60] [-lr 0.06]\n"
   <<"  bytefast-fusion predict BASE.bin WORDS.lex WEIGHTS.fuse INPUT.txt [-gateMargin 0.75]\n"
   <<"  bytefast-fusion bench BASE.bin WORDS.lex WEIGHTS.fuse INPUT.txt [-gateMargin 0.75]\n";
}
} // namespace

int main(int argc,char** argv) {
 try {
  if(argc<6){usageFusion();return 2;}
  const std::string command=argv[1];
  Model model;model.load(argv[2]);
  Lexicon lex;lex.load(model,argv[3]);
  Fusion fuse;
  if(command=="fit") {
    uint32_t epochs=60;
    double lr=0.06;
    for(int i=6;i<argc;i+=2) {
      if(i+1>=argc)throw std::runtime_error("Missing training option");
      const std::string f=argv[i];
      if(f=="-epochs")epochs=std::stoul(argv[i+1]);
      else if(f=="-lr")lr=std::stod(argv[i+1]);
      else throw std::runtime_error("Unknown option "+f);
    }
    if(epochs<1 || epochs>250 || !std::isfinite(lr) || lr<=0 || lr>1)
      throw std::runtime_error("Invalid fusion training settings");
    const auto examples=readRecords(argv[4]);
    fuse.train(model,lex,examples,epochs,lr);
    fuse.save(argv[5]);
    std::cout<<"FUSION_SAVED "<<argv[5]<<" labels="<<fuse.count()<<'\n';
  } else if(command=="predict"||command=="bench") {
    if(argc<7){usageFusion();return 2;}
    fuse.load(model,lex,argv[4]);
    double gate=2.;
    for(int i=7;i<argc;i+=2) {
      if(i+1>=argc)throw std::runtime_error("Missing inference option");
      if(std::string(argv[i])=="-gateMargin")gate=std::stod(argv[i+1]);
      else throw std::runtime_error("Unknown inference option");
    }
    if(!std::isfinite(gate)||gate<0||gate>2)
      throw std::runtime_error("Invalid gate");
    std::ifstream in(argv[5]);
    if(!in)throw std::runtime_error("Could not open text file");
    std::string line;uint64_t n=0;
    const auto start=std::chrono::steady_clock::now();
    while(std::getline(in,line)) {
      if(!line.empty()&&line.back()=='\r')line.pop_back();
      const auto pred=fuse.predict(model,lex,line,static_cast<float>(gate));
      if(command=="predict")std::cout<<"__label__"<<model.labels[pred]<<'\n';
      ++n;
    }
    const auto seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-start).count();
    if(command=="bench")std::cout<<"documents="<<n<<" seconds="<<seconds
      <<" docs_per_second="<<n/seconds<<'\n';
  } else {usageFusion();return 2;}
  return 0;
 } catch(const std::exception& e) {
  std::cerr<<"Fusion error: "<<e.what()<<'\n';return 1;
 }
}
