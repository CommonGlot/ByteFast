#include "bytefast_lexical.h"
namespace {
void help() {
  std::cerr<<"Usage:\n"
    <<" bytefast-lexical build BASE_MODEL TRAIN_DATA OUTPUT.lex [maxWords] [minCount]\n"
    <<" bytefast-lexical predict BASE_MODEL LEXICON INPUT [-weight FLOAT] [-gateMargin 0..2]\n"
    <<" bytefast-lexical bench BASE_MODEL LEXICON INPUT [-weight FLOAT]\n";
}
} // namespace

int main(int argc,char** argv) {
  try {
    if (argc<5) {help();return 2;}
    const std::string cmd=argv[1];
    Model model;model.load(argv[2]);
    Lexicon lex;
    if(cmd=="build") {
      const size_t cap=argc>5?std::stoul(argv[5]):50000;
      const uint32_t count=argc>6?std::stoul(argv[6]):3;
      if(cap>1000000 || !count) throw std::runtime_error("Invalid lexical budget");
      lex.train(model,readRecords(argv[3]),cap,count);
      lex.save(argv[4]);
    } else if(cmd=="predict" || cmd=="bench") {
      if(argc<5) {help();return 2;}
      lex.load(model,argv[3]);
      float weight=1.0f,gateMargin=2.0f;
      for(int i=5;i<argc;i+=2) {
        if(i+1>=argc) throw std::runtime_error("Missing lexical option value");
        const std::string option=argv[i];
        if(option=="-weight") weight=std::stof(argv[i+1]);
        else if(option=="-gateMargin") gateMargin=std::stof(argv[i+1]);
        else throw std::runtime_error("Unknown option: "+option);
      }
      if(!std::isfinite(weight) || weight<0 || weight>10 ||
         !std::isfinite(gateMargin) || gateMargin<0 || gateMargin>2)
        throw std::runtime_error("Invalid lexical weight/gate");
      std::ifstream f(argv[4]);
      if(!f) throw std::runtime_error("Missing input file");
      std::string line;size_t total=0;
      const auto start=std::chrono::steady_clock::now();
      while(std::getline(f,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        const auto prediction=lex.predict(model,line,weight,gateMargin);
        if(cmd=="predict") std::cout<<"__label__"<<model.labels[prediction]<<'\n';
        ++total;
      }
      if(cmd=="bench") {
        const auto sec=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-start).count();
        std::cout<<"documents="<<total<<" seconds="<<sec
                 <<" docs_per_second="<<total/sec<<'\n';
      }
    } else {help();return 2;}
    return 0;
  } catch (const std::exception& e) {
    std::cerr<<"Lexical error: "<<e.what()<<'\n';return 1;
  }
}
