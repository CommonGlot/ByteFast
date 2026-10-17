#include "bytefast_calibrated.h"
namespace {
void calibratedUsage() {
  std::cerr<<"Usage:\n"
    <<"  bytefast-calibrated fit BASE WORDS CALIBRATION.fasttext VALIDATION.fasttext OUTPUT.fuse"
      " [-epochs N] [-lr FLOAT] [-l2 FLOAT] [-patience N]\n"
    <<"  bytefast-calibrated predict BASE WORDS FUSION MODEL_INPUT.txt [-gate 0..2]\n"
    <<"  bytefast-calibrated bench BASE WORDS FUSION MODEL_INPUT.txt [-gate 0..2]\n";
}
} // namespace

int main(int argc,char** argv) {
  try {
    if(argc<6){calibratedUsage();return 2;}
    const std::string command=argv[1];
    Model base;base.load(argv[2]);
    Lexicon lex;lex.load(base,argv[3]);
    Fusion fusion;
    if(command=="fit") {
      if(argc<7||(argc-7)%2!=0) {calibratedUsage();return 2;}
      FusionOptions options;
      for(int i=7;i<argc;i+=2) {
        const std::string flag=argv[i];
        if(flag=="-epochs")options.maxEpochs=static_cast<unsigned>(std::stoul(argv[i+1]));
        else if(flag=="-lr")options.step=std::stod(argv[i+1]);
        else if(flag=="-l2")options.l2=std::stod(argv[i+1]);
        else if(flag=="-patience")options.patience=static_cast<unsigned>(std::stoul(argv[i+1]));
        else throw std::runtime_error("Unknown fit option: "+flag);
      }
      fusion.fit(base,lex,readRecords(argv[4]),readRecords(argv[5]),options);
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
    }else{calibratedUsage();return 2;}
    return 0;
  }catch(const std::exception& e){
    std::cerr<<"Calibrated lexical error: "<<e.what()<<'\n';return 1;
  }
}
