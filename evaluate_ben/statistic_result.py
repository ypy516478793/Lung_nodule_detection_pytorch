import pandas as pd
import matplotlib.pyplot as plt

# ## with fine-tuning
# csv_path = "./detector_ben/results/methodist_finetuned_mode3/froc_gt_prob_vectors_mode_withft_detp0.0_nms0.1.csv"
# result = pd.read_csv(csv_path, header=None)
#
# pos_ids = result[0] == 1
# neg_ids = result[0] == 0
#
# pos_probs = [i for i in result[1][pos_ids] if i > 0]
# neg_probs = result[1][neg_ids].tolist()
#
# plt.hist(pos_probs, bins=len(pos_probs), label="{:d}_TP".format(len(pos_probs)))
# plt.hist(neg_probs, bins=len(neg_probs), label="{:d}_FP".format(len(neg_probs)))
# plt.legend()
# plt.title("mode normalization3 pretrained on LUNA with finetuning")
# plt.show()
#
# ## without fine-tuning
# csv_path = "./detector_ben/results/methodist_pretrainedLUNA_mode3/froc_gt_prob_vectors_mode_noft_detp0.0_nms0.1.csv"
# result = pd.read_csv(csv_path, header=None)
#
# pos_ids = result[0] == 1
# neg_ids = result[0] == 0
#
# pos_probs = [i for i in result[1][pos_ids] if i > 0]
# neg_probs = result[1][neg_ids].tolist()
#
# plt.hist(pos_probs, bins=len(pos_probs), label="{:d}_TP".format(len(pos_probs)))
# plt.hist(neg_probs, bins=len(neg_probs), label="{:d}_FP".format(len(neg_probs)))
# plt.legend()
# plt.title("mode normalization3 pretrained on LUNA without finetuning")
# plt.show()



## minmax fine-tuning old lung mask
csv_path = "./detector_ben/results/methodist_finetuned_minmax/froc_gt_prob_vectors_ft_detp0.0_nms0.1.csv"
result = pd.read_csv(csv_path, header=None)

pos_ids = result[0] == 1
neg_ids = result[0] == 0

pos_probs = [i for i in result[1][pos_ids] if i > 0]
neg_probs = result[1][neg_ids].tolist()

plt.hist(pos_probs, bins=len(pos_probs), label="{:d}_TP".format(len(pos_probs)))
plt.hist(neg_probs, bins=len(neg_probs), label="{:d}_FP".format(len(neg_probs)))
plt.legend()
plt.title("minmax fine-tuning old lung mask")
plt.show()

## minmax fine-tuning new lung mask
csv_path = "./detector_ben/results/methodist_finetuned_minmax_newLungSeg/froc_gt_prob_vectors_ft_detp0.0_nms0.1.csv"
result = pd.read_csv(csv_path, header=None)

pos_ids = result[0] == 1
neg_ids = result[0] == 0

pos_probs = [i for i in result[1][pos_ids] if i > 0]
neg_probs = result[1][neg_ids].tolist()

plt.hist(pos_probs, bins=len(pos_probs), label="{:d}_TP".format(len(pos_probs)))
plt.hist(neg_probs, bins=len(neg_probs), label="{:d}_FP".format(len(neg_probs)))
plt.legend()
plt.title("minmax fine-tuning new lung mask")
plt.show()


print("")