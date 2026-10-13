// Lightweight supervised score-fusion layer for ByteFast's sparse lexical model.
// Trains the importance of byte evidence and word evidence on calibration data.
// The raw UTF-8 byte encoder, lexical vocabulary, and original fastText remain
// unchanged. CPU C++17, MIT License.
#include "bytefast_lexical.h"

namespace {
struct Observation {
  std::vector<float> byteLog, word;
  uint32_t gold;
};

struct Fusion {
  std::vector<std::string> labels;
  // byte scale, global lexical scale, per-label lexical offsets, per-label bias
  std::vector<double> theta;
  size_t nlabels() const { return labels.size(); }

  void init(const Model& base) {
    labels=base.labels;
    theta.assign(2+2*nlabels(),0.);
    theta[0]=1.;
    theta[1]=1.;
  }
  Observation encode(const Model& base,const Lexicon& lex,
                     const Record& r,uint32_t gold) const {
    const auto p=base.probabilities(
      base.embedding(base.features(r.text)));
    Observation obs;
    obs.gold=gold;
    obs.byteLog.resize(p.size());
    for(size_t i=0;i<p.size();++i)
      obs.byteLog[i]=std::log(std::max(p[i],1e-20f));
    obs.word=lex.evidence(r.text);
    return obs;
  }
  std::vector<Observation> encodeFile(const Model& base,const Lexicon& lex,
                                     const std::vector<Record>& records) const {
    std::unordered_map<std::string,uint32_t> labelIds;
    for(uint32_t c=0;c<nlabels();++c)labelIds[labels[c]]=c;
    std::vector<Observation> out;out.reserve(records.size());
    for(const auto& rec:records) {
      const auto it=labelIds.find(rec.label);
      if(it!=labelIds.end())out.push_back(encode(base,lex,rec,it->second));
    }
    if(out.empty())throw std::runtime_error("No matching calibration labels");
    return out;
  }
  void logits(const Observation& o,std::vector<double>& v) const {
    const size_t C=nlabels();
    v.resize(C);
    for(size_t c=0;c<C;++c) {
      v[c]=theta[0]*o.byteLog[c]+
            (theta[1]+theta[2+c])*o.word[c]+theta[2+C+c];
    }
  }
  static double probabilities(std::vector<double>& s,uint32_t gold) {
    const double largest=*std::max_element(s.begin(),s.end());
    double total=0.;
    for(double& v:s){v=std::exp(v-largest);total+=v;}
    const double loss=std::log(total)+largest-
      (largest+std::log(std::max(s[gold],1e-300)));
    for(double& v:s)v/=total;
    return loss;
  }
  double evaluateCE(const std::vector<Observation>& examples,
                    double* accuracy=nullptr) const {
    double loss=0.;uint64_t correct=0;
    std::vector<double> v;
    for(const auto& e:examples) {
      logits(e,v);
      const uint32_t best=static_cast<uint32_t>(
        std::max_element(v.begin(),v.end())-v.begin());
      correct+=(best==e.gold);
      loss+=probabilities(v,e.gold);
    }
    if(accuracy)*accuracy=double(correct)/examples.size();
    return loss/examples.size();
  }
  void fit(const Model& base,const Lexicon& lex,
           const std::vector<Record>& train,
           const std::vector<Record>& dev,
           unsigned maxEpochs=90) {
    init(base);
    const auto fitted=encodeFile(base,lex,train);
    const auto validation=encodeFile(base,lex,dev);
    std::cerr<<"CALIBRATION_COUNTS train="<<fitted.size()
             <<" validation="<<validation.size()
             <<" labels="<<labels.size()<<'\n';
    std::vector<double> first(theta.size(),0),second(theta.size(),0);
    auto best=theta;
    double bestLoss=evaluateCE(validation),trainAccuracy;
    unsigned bestEpoch=0,noImprove=0;
    std::vector<double> v,gradient(theta.size());
    for(unsigned epoch=1;epoch<=maxEpochs;++epoch) {
      std::fill(gradient.begin(),gradient.end(),0.);
      const size_t C=nlabels();
      for(const auto& e:fitted) {
        logits(e,v);
        probabilities(v,e.gold);
        for(size_t c=0;c<C;++c) {
          const double err=v[c]-(c==e.gold?1.:0.);
          gradient[0]+=err*e.byteLog[c];
          gradient[1]+=err*e.word[c];
          gradient[2+c]+=err*e.word[c];
          gradient[2+C+c]+=err;
        }
      }
      for(double& g:gradient)g/=fitted.size();
      // Keep rare-label corrections from overfitting tiny calibration sets.
      for(size_t c=0;c<C;++c) {
        gradient[2+c]+=0.01*theta[2+c]/C;
        gradient[2+C+c]+=0.01*theta[2+C+c]/C;
      }
      const double step=0.07;
      for(size_t i=0;i<theta.size();++i) {
        first[i]=.9*first[i]+.1*gradient[i];
        second[i]=.999*second[i]+.001*gradient[i]*gradient[i];
        const double m=first[i]/(1-std::pow(.9,epoch));
        const double var=second[i]/(1-std::pow(.999,epoch));
        theta[i]-=step*m/(std::sqrt(var)+1e-8);
      }
      theta[0]=std::clamp(theta[0],0.05,5.);
      theta[1]=std::clamp(theta[1],0.,10.);
      for(size_t c=0;c<C;++c) {
        theta[2+c]=std::clamp(theta[2+c],-5.,5.);
        theta[2+C+c]=std::clamp(theta[2+C+c],-4.,4.);
      }
      double valAccuracy=0;
      const double validationLoss=evaluateCE(validation,&valAccuracy);
      if(validationLoss<bestLoss-1e-5) {
        bestLoss=validationLoss;
        best=theta;
        bestEpoch=epoch;
        noImprove=0;
      }else ++noImprove;
      if(epoch==1||epoch%10==0||noImprove>=16) {
        std::cerr<<"CALIBRATION_EPOCH epoch="<<epoch
                 <<" validation_ce="<<validationLoss
                 <<" validation_accuracy="<<valAccuracy
                 <<" byte_coefficient="<<theta[0]
                 <<" word_coefficient="<<theta[1]<<'\n';
      }
      if(noImprove>=16)break;
    }
    theta=std::move(best);
    double minWord=1e50,maxWord=-1e50;
    for(size_t c=0;c<nlabels();++c) {
      const double w=theta[1]+theta[2+c];
      minWord=std::min(minWord,w);
      maxWord=std::max(maxWord,w);
    }
    std::cerr<<"CALIBRATION_FINAL epoch="<<bestEpoch
             <<" heldout_validation_ce="<<bestLoss
             <<" learned_byte_coefficient="<<theta[0]
             <<" learned_word_global_coefficient="<<theta[1]
             <<" learned_word_per_label_min="<<minWord
             <<" learned_word_per_label_max="<<maxWord<<'\n';
  }

  uint32_t predict(const Model& base,const Lexicon& lex,
                   const std::string& text,double gate) const {
    const auto p=base.probabilities(base.embedding(base.features(text)));
    uint32_t best=0;float first=-1,second=-1;
    for(uint32_t c=0;c<p.size();++c) {
      if(p[c]>first) {
        second=first;first=p[c];best=c;
      }else if(p[c]>second)second=p[c];
    }
    if(first-second>=gate)return best;
    const auto ev=lex.evidence(text);
    double bestScore=-1e300;
    for(size_t c=0;c<p.size();++c) {
      const double score=
        theta[0]*std::log(std::max(1e-20f,p[c]))+
        (theta[1]+theta[2+c])*ev[c]+theta[2+nlabels()+c];
      if(score>bestScore){bestScore=score;best=static_cast<uint32_t>(c);}
    }
    return best;
  }

  void save(const std::string& path) const {
    std::ofstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Could not write learned fusion");
    Model::write(f,uint32_t{0x42464c58U});
    Model::write(f,uint32_t{1});
    Model::write(f,static_cast<uint32_t>(nlabels()));
    for(const auto& s:labels) {
      Model::write(f,static_cast<uint32_t>(s.size()));f.write(s.data(),s.size());
    }
    for(double w:theta) {
      const float f32=static_cast<float>(w);
      Model::write(f,f32);
    }
    if(!f)throw std::runtime_error("Failed writing learned fusion");
  }
  void load(const Model& base,const Lexicon& lex,const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Cannot read learned fusion");
    uint32_t magic,version,n;
    Model::read(f,magic);Model::read(f,version);Model::read(f,n);
    if(magic!=0x42464c58U||version!=1||n!=base.labels.size())
      throw std::runtime_error("Bad fusion model header");
    labels.clear();
    for(uint32_t i=0;i<n;++i) {
      uint32_t len;Model::read(f,len);
      if(len>256)throw std::runtime_error("Bad label in fusion model");
      std::string s(len,'\0');
      if(len&&!f.read(&s[0],len))throw std::runtime_error("Truncated label");
      labels.push_back(std::move(s));
    }
    if(labels!=base.labels||labels!=lex.labels)
      throw std::runtime_error("Fusion labels do not match ByteFast model");
    theta.resize(2+2*n);
    for(double& v:theta) {
      float tmp;Model::read(f,tmp);
      if(!std::isfinite(tmp)||std::abs(tmp)>100)
        throw std::runtime_error("Corrupted fusion coefficient");
      v=tmp;
    }
  }
};
void usage() {
  std::cerr<<"Usage:\n"
    <<"  bytefast-calibrated fit BASE WORDS CALIBRATION.fasttext VALIDATION.fasttext OUTPUT.fuse\n"
    <<"  bytefast-calibrated predict BASE WORDS FUSION MODEL_INPUT.txt [-gate 0..2]\n"
    <<"  bytefast-calibrated bench BASE WORDS FUSION MODEL_INPUT.txt [-gate 0..2]\n";
}
} // namespace

int main(int argc,char** argv) {
  try {
    if(argc<6){usage();return 2;}
    const std::string command=argv[1];
    Model base;base.load(argv[2]);
    Lexicon lex;lex.load(base,argv[3]);
    Fusion fusion;
    if(command=="fit") {
      if(argc!=7) {usage();return 2;}
      fusion.fit(base,lex,readRecords(argv[4]),readRecords(argv[5]));
      fusion.save(argv[6]);
    }else if(command=="predict"||command=="bench") {
      fusion.load(base,lex,argv[4]);
      double gate=2.;
      if(argc>6) {
        if(argc!=8||std::string(argv[6])!="-gate")
          throw std::runtime_error("Expected -gate VALUE");
        gate=std::stod(argv[7]);
      }
      if(!std::isfinite(gate)||gate<0||gate>2)
        throw std::runtime_error("Gate must be between 0 and 2");
      std::ifstream input(argv[5]);
      if(!input)throw std::runtime_error("Cannot open text file");
      std::string line;size_t n=0;
      const auto start=std::chrono::steady_clock::now();
      while(std::getline(input,line)) {
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        const auto pred=fusion.predict(base,lex,line,gate);
        if(command=="predict")
          std::cout<<"__label__"<<fusion.labels[pred]<<'\n';
        ++n;
      }
      if(command=="bench") {
        const auto seconds=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-start).count();
        std::cout<<"documents="<<n<<" seconds="<<seconds
                 <<" docs_per_second="<<n/seconds<<'\n';
      }
    }else{usage();return 2;}
    return 0;
  }catch(const std::exception& e){
    std::cerr<<"Calibrated lexical error: "<<e.what()<<'\n';return 1;
  }
}
