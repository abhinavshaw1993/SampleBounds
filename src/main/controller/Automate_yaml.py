import yaml
from bounds import BoundsExperiment
path="C:/Users/Shruti Jadon/Desktop/IndependentStudies/IS/SampleBounds/src/main/resources/"
with open(path+"config.yml") as f:
    list_doc = yaml.load(f)

distribution=["normal","uniform","exponential","beta","mix_gauss_2_sym_uni","mix_gauss_2_sym_multi","mix_gauss_2_nonsym_uni","mix_gauss_2_nonsym_multi","mix_gauss_4_sym_uni","mix_gauss_4_sym_multi","mix_gauss_4_nonsym_uni","mix_gauss_4_nonsym_multi","student_t_3","student_t_5","double_exponential","mix_double_exponential"]
for dis in distribution:
    list_doc["sample_statistics"]["distribution"]=dis
    with open(path + "config.yml", "w") as f:
        yaml.dump(list_doc, f)
    B = BoundsExperiment(algo=["ORDSTAT","CLT","CHERNOFF_HOEFFDING","MASSART"])
    B.run_experiments()
    print (list_doc["sample_statistics"]["distribution"] +" done")
