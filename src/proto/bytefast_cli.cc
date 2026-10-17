#include "bytefast_model.h"
int main(int argc,char** argv) {
 try {
  if (argc<4) {usage();return 2;}
  std::string cmd=argv[1];
  Model model;
  if (cmd=="train") {
    for (int i=4;i<argc;i+=2) {
      if (i+1>=argc) throw std::runtime_error("Missing option value");
      std::string flag=argv[i];
      if (flag=="-bucket") model.opt.bucket=numeric(argv[i+1]);
      else if(flag=="-dim") model.opt.dim=numeric(argv[i+1]);
      else if(flag=="-epoch") model.opt.epochs=numeric(argv[i+1]);
      else if(flag=="-lr") model.opt.lr=std::stof(argv[i+1]);
      else if(flag=="-minn") model.opt.minN=numeric(argv[i+1]);
      else if(flag=="-maxn") model.opt.maxN=numeric(argv[i+1]);
      else if(flag=="-hashCount") model.opt.hashes=numeric(argv[i+1]);
      else if(flag=="-positionBins") {
        const auto bins=numeric(argv[i+1]);
        if (bins!=0 && bins!=4) throw std::runtime_error("positionBins must be 0 or 4");
        model.opt.positions=bins?1:0;
      }
      else if(flag=="-maxBytes") model.opt.maxBytes=numeric(argv[i+1]);
      else if(flag=="-asciiFold") model.opt.asciiFold=numeric(argv[i+1]);
      else throw std::runtime_error("Unknown option: "+flag);
    }
    if (!model.opt.bucket || !model.opt.dim || !model.opt.minN ||
        model.opt.maxN>16 || model.opt.minN>model.opt.maxN ||
        !model.opt.hashes || model.opt.hashes>4 || !model.opt.maxBytes ||
        !model.opt.epochs || model.opt.asciiFold>1 || !(model.opt.lr>0))
      throw std::runtime_error("Invalid hyperparameters");
    const auto records=readRecords(argv[2]);
    model.train(records);
    model.save(argv[3]);
    std::cerr << "saved=" << argv[3] << " labels=" << model.labels.size() << '\n';
  } else if(cmd=="test") {
    model.load(argv[2]);
    auto records=readRecords(argv[3]);
    uint64_t correct=0;
    for (const auto& r:records) correct+=(model.labels[model.predict(r.text)]==r.label);
    std::cout << "accuracy=" << double(correct)/records.size()
              << " correct=" << correct << " total=" << records.size() << '\n';
  } else if(cmd=="predict" || cmd=="predict-scores" || cmd=="bench") {
    float threshold=0.0f, minMargin=0.0f;
    for (int i=4;i<argc;i+=2) {
      if (i+1>=argc) throw std::runtime_error("Missing inference option");
      const std::string flag=argv[i];
      if(flag=="-rejectThreshold") threshold=std::stof(argv[i+1]);
      else if(flag=="-marginThreshold") minMargin=std::stof(argv[i+1]);
      else throw std::runtime_error("Unknown inference option: "+flag);
    }
    if(!std::isfinite(threshold) || threshold<0 || threshold>1 ||
       !std::isfinite(minMargin) || minMargin<0 || minMargin>1)
      throw std::runtime_error("Rejection thresholds must be in [0,1]");
    model.load(argv[2]);
    std::ifstream f(argv[3]);
    if (!f) throw std::runtime_error("Cannot open input");
    std::string s;
    size_t total=0;
    const auto start=std::chrono::steady_clock::now();
    while (std::getline(f,s)) {
      if (!s.empty() && s.back()=='\r') s.pop_back();
      float confidence=0, margin=0;
      const auto pred=model.predict(s,&confidence,&margin);
      ++total;
      if(cmd=="predict" || cmd=="predict-scores") {
        const bool reject=confidence<threshold || margin<minMargin || s.empty();
        std::cout << "__label__" << (reject ? "und" : model.labels[pred])
                  << " " << std::setprecision(6) << confidence;
        if(cmd=="predict-scores") std::cout << " " << margin;
        std::cout << '\n';
      }
    }
    const auto seconds=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-start).count();
    if (cmd=="bench")
      std::cout << "documents=" << total << " seconds=" << seconds
                << " docs_per_second=" << total/seconds << '\n';
  } else { usage();return 2; }
  return 0;
 } catch(const std::exception& e) {
   std::cerr << "ByteFast error: " << e.what() << '\n';return 1;
 }
}
