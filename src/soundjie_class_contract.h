#ifndef SOUNDJIE_CLASS_CONTRACT_H
#define SOUNDJIE_CLASS_CONTRACT_H
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace soundjie {
inline unsigned physical_class_count(const std::string& name) {
  if(name=="soundjie-alert5bg-v3")return 6;
  if(name=="soundjie-alert4bg-v2")return 5;
  if(name=="soundjie-siren5-v1")return 6;
  if(name=="soundjie-legacy8-v1")return 8;
  throw std::invalid_argument("unknown class contract");
}
inline bool alert4_contract(const std::string& name) { physical_class_count(name);return name=="soundjie-alert4bg-v2"; }
inline bool alert5_contract(const std::string& name) { physical_class_count(name);return name=="soundjie-alert5bg-v3"; }
inline unsigned logical_class_count(const std::string& name) {
  return name=="soundjie-siren5-v1"?5:physical_class_count(name);
}
inline bool siren5_contract(const std::string& name) {
  if(name=="soundjie-siren5-v1") return true;
  if(name=="soundjie-legacy8-v1"||name=="soundjie-alert4bg-v2"||name=="soundjie-alert5bg-v3") return false;
  throw std::invalid_argument("unknown class contract");
}
inline const char* class_name(int id,bool five);
inline const char* class_name(int id,const std::string& contract) {
  if(alert5_contract(contract)) {
    static const char* labels[]={"警笛","车辆鸣笛","门铃","狗叫","普通说话声","背景"};
    return id>=0&&id<6?labels[id]:"不可用";
  }
  if(alert4_contract(contract)) {
    static const char* labels[]={"警笛","车辆鸣笛","门铃","狗叫","背景"};
    return id>=0&&id<5?labels[id]:"不可用";
  }
  return class_name(id,siren5_contract(contract));
}
inline int alert4_decision(const std::vector<double>& scores,const std::vector<double>& thresholds) {
  if(scores.size()!=5||thresholds.size()!=4)throw std::runtime_error("alert4 decision contract mismatch");
  const int top=std::max_element(scores.begin(),scores.end())-scores.begin();
  return top<4&&scores[top]<thresholds[top]?4:top;
}
inline int alert5_decision(const std::vector<double>& scores,const std::vector<double>& thresholds) {
  if(scores.size()!=6||thresholds.size()!=5)throw std::runtime_error("alert5 decision contract mismatch");
  const int top=std::max_element(scores.begin(),scores.end())-scores.begin();
  return top<5&&scores[top]<thresholds[top]?5:top;
}
inline const char* class_name(int id,bool five) {
  static const char* labels5[]={"警笛","车辆鸣笛","玻璃破碎","敲门/门铃","背景"};
  static const char* labels8[]={"火灾或烟雾报警器","应急车辆警笛","汽车鸣笛","玻璃破碎","敲门或门铃","紧急人声","普通说话声","背景或未知声音"};
  return id<0||id>=(five?5:8)?"不可用":five?labels5[id]:labels8[id];
}
inline std::vector<double> merge_siren_probabilities(const std::vector<double>& six) {
  if(six.size()!=6)throw std::runtime_error("six probabilities required");
  double sum=0;
  for(double p:six){if(!std::isfinite(p)||p<0||p>1)throw std::runtime_error("invalid probability");sum+=p;}
  if(std::abs(sum-1)>1e-6)throw std::runtime_error("probabilities not normalized");
  return {six[0]+six[1],six[2],six[3],six[4],six[5]};
}
inline int siren_decision(const std::vector<double>& five) {
  if(five.size()!=5)throw std::runtime_error("five probabilities required");
  const int top=std::max_element(five.begin(),five.end())-five.begin();
  static const double thresholds[]={.4,.6500000000000002,.4,.4};
  return top<4&&five[top]<thresholds[top]?4:top;
}
}
#endif
