// Experimental supervised word-embedding branch for ByteFast.
// Byte 2-5-grams remain untouched and whitespace-free; optional word features
// provide additional learned lexical discrimination for nearby languages.
// Pure C++17, CPU only, dependency free. MIT license.
#include "bytefast_lexical.h"
#include <unordered_set>
#include <map>
#include <sstream>
#include <cstring>

namespace {
struct DocStats {
  uint32_t total=0;
  std::unordered_map<uint32_t,uint32_t> byLanguage;
};
struct Candidate {
  std::string word;
  double score=0;
};
struct TrainedExample {
  std::vector<uint32_t> wordIds;
  std::vector<float> byteLogP;
  uint32_t label=0;
};
uint64_t pairKey(uint32_t a,uint32_t b) {
  if(a>b)std::swap(a,b);
  return (uint64_t(a)<<32)|b;
}
uint32_t asUnsigned(const std::string& value) {
  size_t consumed=0;
  auto n=std::stoul(value,&consumed);
  if(consumed!=value.size()||n>1000000000UL)
    throw std::runtime_error("Invalid integer: "+value);
  return static_cast<uint32_t>(n);
}
class WordBranch {
 public:
  std::vector<std::string> labels, tokens;
  std::unordered_map<std::string,uint32_t> wordId;
  std::vector<float> embed,head,bias;
  uint32_t dim=16,maxBytes=512;
  float byteWeight=1.0f;
  std::string selection="confusion";

  void selectWords(const Model& base,const std::vector<Record>& docs,
                   size_t cap,uint32_t minCount,const std::string& ranking) {
    labels=base.labels;
    maxBytes=base.opt.maxBytes;
    selection=ranking;
    std::unordered_map<std::string,uint32_t> langId;
    for(uint32_t i=0;i<labels.size();++i)langId[labels[i]]=i;
    std::vector<uint32_t> totals(labels.size(),0);
    std::unordered_map<std::string,DocStats> counts;
    std::unordered_map<uint64_t,uint32_t> pairErrors;
    for(const auto& record:docs) {
      auto it=langId.find(record.label);
      if(it==langId.end())continue;
      const uint32_t lid=it->second;
      ++totals[lid];
      if(ranking=="confusion") {
        const uint32_t predicted=base.predict(record.text);
        if(predicted!=lid)++pairErrors[pairKey(lid,predicted)];
      }
      const auto pieces=words(record.text,maxBytes);
      std::unordered_set<std::string> unique(pieces.begin(),pieces.end());
      for(const auto& word:unique) {
        auto& stat=counts[word];
        ++stat.total;
        ++stat.byLanguage[lid];
      }
    }
    std::vector<std::pair<uint64_t,uint32_t>> confusions(
      pairErrors.begin(),pairErrors.end());
    std::sort(confusions.begin(),confusions.end(),
      [](const auto& a,const auto& b) {
        if(a.second!=b.second)return a.second>b.second;
        return a.first<b.first;
      });
    // Rank only pairs with repeated errors so isolated mistakes cannot
    // dominate the vocabulary. Keep the 48 worst pairs.
    if(confusions.size()>48)confusions.resize(48);
    size_t eligible=0;
    std::vector<Candidate> candidates;
    candidates.reserve(counts.size());
    for(const auto& kv:counts) {
      const auto& w=kv.first;const auto& s=kv.second;
      if(s.total<minCount)continue;
      ++eligible;
      uint32_t nlang=0;
      for(const auto& entry:s.byLanguage)
        if(entry.second>=2)++nlang;
      const double languageIdf=std::log(
        (double(labels.size())+1)/(double(nlang)+1));
      const double tf=std::log1p(static_cast<double>(s.total));
      double benefit=0;
      if(ranking=="confusion") {
        for(const auto& bad:confusions) {
          const auto first=uint32_t(bad.first>>32);
          const auto second=uint32_t(bad.first&0xffffffffu);
          const auto one=s.byLanguage.find(first);
          const auto two=s.byLanguage.find(second);
          const double p1=(double(one==s.byLanguage.end()?0:one->second)+0.5)/
                          (totals[first]+1.0);
          const double p2=(double(two==s.byLanguage.end()?0:two->second)+0.5)/
                          (totals[second]+1.0);
          // Strong tokens distinguish repeated genuine label confusions.
          // The sqrt avoids allowing just one dominant pair to take over.
          const double strength=std::min(4.,std::abs(std::log(p1/p2)));
          benefit=std::max(benefit,strength*
                  std::min(1.,std::sqrt(bad.second/20.)));
        }
      }
      // TF-IDF is always retained; only the second factor changes.
      const double score=tf*(0.25+languageIdf)*
                           (ranking=="confusion"?(0.5+benefit):1.);
      candidates.push_back({w,score});
    }
    auto bestFirst=[](const Candidate& a,const Candidate& b) {
      if(a.score!=b.score)return a.score>b.score;
      return a.word<b.word;
    };
    if(candidates.size()>cap) {
      std::nth_element(candidates.begin(),candidates.begin()+cap,
                       candidates.end(),bestFirst);
      candidates.resize(cap);
    }
    std::sort(candidates.begin(),candidates.end(),bestFirst);
    for(auto& v:candidates) {
      wordId[v.word]=static_cast<uint32_t>(tokens.size());
      tokens.push_back(std::move(v.word));
    }
    std::cerr<<"WORDEMB_VOCAB rank="<<selection<<" cap="<<cap
             <<" eligible="<<eligible<<" kept="<<tokens.size()
             <<" hard_pairs="<<confusions.size()<<'\n';
    for(size_t i=0;i<std::min<size_t>(6,confusions.size());++i) {
      const auto a=uint32_t(confusions[i].first>>32);
      const auto b=uint32_t(confusions[i].first&0xffffffffu);
      std::cerr<<"WORDEMB_CONFUSION "<<labels[a]<<" "<<labels[b]
               <<" count="<<confusions[i].second<<'\n';
    }
  }

  std::vector<uint32_t> lookup(const std::string& text) const {
    const auto pieces=words(text,maxBytes);
    std::vector<uint32_t> ids;
    ids.reserve(pieces.size());
    for(const auto& w:pieces) {
      auto it=wordId.find(w);
      if(it!=wordId.end())ids.push_back(it->second);
    }
    std::sort(ids.begin(),ids.end());
    ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
    return ids;
  }
  TrainedExample encode(const Model& base,const Record& r,
                        uint32_t label) const {
    TrainedExample ex;
    ex.label=label;
    ex.wordIds=lookup(r.text);
    const auto probs=base.probabilities(base.embedding(base.features(r.text)));
    ex.byteLogP.reserve(probs.size());
    for(const float p:probs) ex.byteLogP.push_back(
      std::log(std::max(p,1e-20f)));
    return ex;
  }
  std::vector<float> wordVector(const std::vector<uint32_t>& ids) const {
    std::vector<float> result(dim,0);
    if(ids.empty())return result;
    const float norm=1.f/std::sqrt(static_cast<float>(ids.size()));
    for(auto id:ids) {
      const float* ptr=embed.data()+size_t(id)*dim;
      for(uint32_t k=0;k<dim;++k)result[k]+=ptr[k]*norm;
    }
    return result;
  }
  std::vector<float> logits(const std::vector<float>& h,
                            const std::vector<float>& logP) const {
    std::vector<float> value(labels.size());
    for(size_t c=0;c<labels.size();++c) {
      float score=byteWeight*logP[c]+bias[c];
      const float* ptr=head.data()+c*dim;
      for(uint32_t k=0;k<dim;++k)score+=ptr[k]*h[k];
      value[c]=score;
    }
    return value;
  }
  static void softmax(std::vector<float>& s) {
    const float hi=*std::max_element(s.begin(),s.end());
    float total=0;
    for(float& x:s){x=std::exp(x-hi);total+=x;}
    for(float& x:s)x/=total;
  }
  void train(const Model& base,const std::vector<Record>& examples,
             uint32_t epochs,float lr) {
    std::unordered_map<std::string,uint32_t> lmap;
    for(uint32_t i=0;i<labels.size();++i)lmap[labels[i]]=i;
    std::vector<TrainedExample> cache;
    cache.reserve(examples.size());
    for(const auto& e:examples) {
      const auto it=lmap.find(e.label);
      if(it!=lmap.end())cache.push_back(encode(base,e,it->second));
    }
    const size_t cells=tokens.size()*size_t(dim);
    if(cells>140000000ULL || 4ULL*(cells+size_t(dim)*labels.size())>=900000000ULL)
      throw std::runtime_error("Word embedding sidecar would exceed memory cap");
    embed.resize(cells);
    head.resize(labels.size()*size_t(dim));
    bias.assign(labels.size(),0);
    std::mt19937 rng(171);
    std::normal_distribution<float> d(0,0.07f);
    for(float& v:embed)v=d(rng);
    for(float& v:head)v=d(rng)*0.5f;
    std::vector<size_t> order(cache.size());
    std::iota(order.begin(),order.end(),0);
    for(uint32_t epoch=0;epoch<epochs;++epoch) {
      std::mt19937 shuffle(1300+epoch);
      std::shuffle(order.begin(),order.end(),shuffle);
      double loss=0;
      uint64_t correct=0,seen=0;
      float alphaGrad=0;
      for(size_t row:order) {
        const auto& e=cache[row];
        if(e.wordIds.empty())continue;
        const auto h=wordVector(e.wordIds);
        auto p=logits(h,e.byteLogP);
        const uint32_t guess=uint32_t(
          std::max_element(p.begin(),p.end())-p.begin());
        correct+=(guess==e.label);
        const float highest=*std::max_element(p.begin(),p.end());
        float expSum=0;
        for(float& x:p){x=std::exp(x-highest);expSum+=x;}
        loss+=std::log(expSum)+highest-
          (byteWeight*e.byteLogP[e.label]+bias[e.label]+
           [&]{
             float s=0;const float* w=head.data()+size_t(e.label)*dim;
             for(uint32_t k=0;k<dim;++k)s+=w[k]*h[k];
             return s;
           }());
        std::vector<float> g(dim,0.0f);
        float a=0;
        for(uint32_t c=0;c<labels.size();++c) {
          const float error=p[c]/expSum-(c==e.label?1.f:0.f);
          float* w=head.data()+size_t(c)*dim;
          a+=error*e.byteLogP[c];
          // Gradient wrt h must use weights *before* update.
          for(uint32_t k=0;k<dim;++k)g[k]+=error*w[k];
          for(uint32_t k=0;k<dim;++k) {
            w[k]-=lr*(error*h[k]+0.00002f*w[k]);
          }
          bias[c]-=lr*error*0.15f;
        }
        // The relative byte-vs-lexical importance is learned, not fixed.
        alphaGrad=0.98f*alphaGrad+0.02f*a;
        byteWeight=std::clamp(byteWeight-lr*0.01f*alphaGrad,0.3f,2.0f);
        const float norm=lr/std::sqrt(float(e.wordIds.size()));
        for(auto id:e.wordIds) {
          float* w=embed.data()+size_t(id)*dim;
          for(uint32_t k=0;k<dim;++k)
            w[k]-=norm*(g[k]+0.00002f*w[k]);
        }
        ++seen;
      }
      std::cerr<<"WORDEMB_EPOCH "<<epoch+1
               <<" cross_entropy="<<loss/std::max<uint64_t>(1,seen)
               <<" train_accuracy="<<double(correct)/std::max<uint64_t>(1,seen)
               <<" byte_weight="<<byteWeight<<" processed="<<seen<<'\n';
    }
  }
  uint32_t predict(const Model& base,const std::string& text) const {
    const auto probabilities=base.probabilities(
      base.embedding(base.features(text)));
    std::vector<float> logP;
    logP.reserve(probabilities.size());
    for(float p:probabilities)logP.push_back(std::log(std::max(p,1e-20f)));
    const auto ids=lookup(text);
    auto s=logits(wordVector(ids),logP);
    return static_cast<uint32_t>(std::max_element(s.begin(),s.end())-s.begin());
  }
  void save(const std::string& file) const {
    std::ofstream f(file,std::ios::binary);
    if(!f)throw std::runtime_error("Cannot write word embeddings");
    Model::write(f,uint32_t{0x42455744u});
    Model::write(f,uint32_t{1});
    const uint32_t nlabels=uint32_t(labels.size());
    Model::write(f,nlabels);
    Model::write(f,uint32_t(tokens.size()));
    Model::write(f,dim);Model::write(f,maxBytes);Model::write(f,byteWeight);
    for(const auto& label:labels) {
      Model::write(f,uint32_t(label.size()));f.write(label.data(),label.size());
    }
    for(const auto& token:tokens) {
      Model::write(f,uint32_t(token.size()));f.write(token.data(),token.size());
    }
    f.write(reinterpret_cast<const char*>(embed.data()),embed.size()*sizeof(float));
    f.write(reinterpret_cast<const char*>(head.data()),head.size()*sizeof(float));
    f.write(reinterpret_cast<const char*>(bias.data()),bias.size()*sizeof(float));
    if(!f)throw std::runtime_error("Failed writing word embedding weights");
  }
  void load(const Model& base,const std::string& file) {
    std::ifstream f(file,std::ios::binary);
    if(!f)throw std::runtime_error("Cannot open word embeddings");
    uint32_t magic,version,nlang,nwords;
    Model::read(f,magic);Model::read(f,version);
    Model::read(f,nlang);Model::read(f,nwords);
    Model::read(f,dim);Model::read(f,maxBytes);Model::read(f,byteWeight);
    if(magic!=0x42455744u||version!=1||!dim||dim>128||nwords>500000||
       nlang!=base.labels.size()||maxBytes!=base.opt.maxBytes ||
       !std::isfinite(byteWeight)||byteWeight<0.1f||byteWeight>3.f||
       4ULL*(uint64_t(nwords)*dim+uint64_t(nlang)*dim+nlang)>900000000ULL)
      throw std::runtime_error("Invalid word-branch dimensions or version");
    auto readString=[&f](uint32_t maxlen) {
      uint32_t n;Model::read(f,n);
      if(n>maxlen)throw std::runtime_error("Invalid string length");
      std::string s(n,'\0');
      if(n&&!f.read(&s[0],n))throw std::runtime_error("Truncated word vocabulary");
      return s;
    };
    for(uint32_t i=0;i<nlang;++i)labels.push_back(readString(256));
    if(labels!=base.labels)throw std::runtime_error("Word model label order differs");
    for(uint32_t i=0;i<nwords;++i) {
      auto w=readString(48);
      if(!wordId.emplace(w,i).second)
        throw std::runtime_error("Duplicate word in word embeddings");
      tokens.push_back(std::move(w));
    }
    embed.resize(size_t(nwords)*dim);
    head.resize(size_t(nlang)*dim);
    bias.resize(nlang);
    if(!f.read(reinterpret_cast<char*>(embed.data()),embed.size()*4)||
       !f.read(reinterpret_cast<char*>(head.data()),head.size()*4)||
       !f.read(reinterpret_cast<char*>(bias.data()),bias.size()*4))
       throw std::runtime_error("Truncated word model parameters");
    if(!std::all_of(embed.begin(),embed.end(),[](float x){return std::isfinite(x);}) ||
       !std::all_of(head.begin(),head.end(),[](float x){return std::isfinite(x);}))
       throw std::runtime_error("Non-finite word weights");
  }
};
void usageWord() {
  std::cerr<<"Usage:\n"
    <<" bytefast-wordemb train BYTE_MODEL TRAIN_LABELED OUTPUT [-vocab 50000] [-dim 16] [-minCount 3] [-epoch 5] [-lr 0.02] [-selection tfidf|confusion]\n"
    <<" bytefast-wordemb predict BYTE_MODEL WORD_MODEL TEXT_FILE\n"
    <<" bytefast-wordemb bench BYTE_MODEL WORD_MODEL TEXT_FILE\n";
}
} // namespace
int main(int argc,char** argv) {
  try {
    if(argc<5){usageWord();return 2;}
    const std::string cmd=argv[1];
    Model base;base.load(argv[2]);
    WordBranch model;
    if(cmd=="train") {
      uint32_t cap=50000,dim=16,minCount=3,epochs=5;
      float lr=0.02f;
      std::string ranking="confusion";
      for(int i=5;i<argc;i+=2) {
        if(i+1>=argc)throw std::runtime_error("Missing option value");
        const std::string flag=argv[i];
        if(flag=="-vocab")cap=asUnsigned(argv[i+1]);
        else if(flag=="-dim")dim=asUnsigned(argv[i+1]);
        else if(flag=="-minCount")minCount=asUnsigned(argv[i+1]);
        else if(flag=="-epoch")epochs=asUnsigned(argv[i+1]);
        else if(flag=="-lr")lr=std::stof(argv[i+1]);
        else if(flag=="-selection")ranking=argv[i+1];
        else throw std::runtime_error("Unknown word-embedding option: "+flag);
      }
      if(cap<1||cap>300000||dim<1||dim>128||minCount<1||epochs<1||
         epochs>40||!std::isfinite(lr)||lr<=0||lr>0.3f||
         (ranking!="tfidf"&&ranking!="confusion"))
        throw std::runtime_error("Invalid word embedding hyperparameters");
      const auto docs=readRecords(argv[3]);
      model.dim=dim;
      model.selectWords(base,docs,cap,minCount,ranking);
      model.train(base,docs,epochs,lr);
      model.save(argv[4]);
      std::cerr<<"WORDEMB_SAVED file="<<argv[4]
               <<" words="<<model.tokens.size()<<" dim="<<dim<<'\n';
    } else if(cmd=="predict"||cmd=="bench") {
      model.load(base,argv[3]);
      std::ifstream input(argv[4]);
      if(!input)throw std::runtime_error("Cannot read text file");
      std::string line;
      size_t n=0;
      const auto start=std::chrono::steady_clock::now();
      while(std::getline(input,line)) {
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        const auto result=model.predict(base,line);
        if(cmd=="predict")std::cout<<"__label__"<<model.labels[result]<<'\n';
        ++n;
      }
      if(cmd=="bench") {
        const auto elapsed=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-start).count();
        std::cout<<"documents="<<n<<" seconds="<<elapsed
                 <<" docs_per_second="<<n/elapsed<<'\n';
      }
    } else{usageWord();return 2;}
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<"ByteFast word embedding error: "<<e.what()<<'\n';
    return 1;
  }
}
