import matplotlib.pyplot as plt
import torch
import numpy as np
import os


def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, run_name, type="encoder"):
        plt.rcParams.update({"lines.linewidth": 2})
        plt.rcParams.update({"lines.markersize": 8})
        plt.rcParams.update({"lines.markeredgewidth": 1})
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.weight"] = "bold"

        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        words = self.tokenizer.decode(padded_sentence).split()
        # print(words)
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        _, attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        ensure_directory_exists(directory_path="./attention_maps")

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            for head_idx in range(attn_map.size(1)):
                att_map = attn_map.squeeze(0).detach().cpu().numpy()
                plot_att = att_map[head_idx]  # Remove batch dimension and convert to NumPy array
                # print(plot_att.shape)

                # Check if the attention probabilities sum to 1 over rows
                total_prob_over_rows = torch.sum(torch.tensor(plot_att), dim=1)
                if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                    print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                    print("Total probability over rows:", total_prob_over_rows.numpy())

                # Create a heatmap of the attention map
                fig, ax = plt.subplots(figsize=(min(len(words) * 0.75, 15), min(len(words) * 0.75, 15)))
                cax = ax.imshow(plot_att, cmap="hot", interpolation="nearest")

                ax.set_xticks(np.arange(len(words)))
                ax.set_yticks(np.arange(len(words)))
                ax.set_xticklabels(words)
                ax.set_yticklabels(words)

                for q in range(len(words)):
                    for k in range(len(words)):
                        text = ax.text(k, q, f"{plot_att[q, k]:.2f}", ha="center", va="center", color="white")

                ax.xaxis.tick_top()  
                fig.colorbar(cax, ax=ax)
                plt.title(f"{run_name.upper()} Attention Map Layer {j + 1} Head {head_idx + 1}", pad=40, fontsize=15)
                plt.tight_layout()
                
                # Save the plot
                if type =="encoder":
                    plt.savefig(f"./attention_maps/{run_name}_attention_map_layer_{j + 1}_head_{head_idx + 1}.png")
                elif type == "decoder":
                    plt.savefig(f"./attention_maps/decoder_attention_map_layer_{j + 1}_head_{head_idx + 1}.png")
                
                # Show the plot
                plt.show()
            


