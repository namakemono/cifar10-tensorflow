import glob
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")

def run():
    is_power_point=False
    save_layers_cmp(is_power_point)
    save_solvers_cmp(is_power_point)
    save_mapping_cmp(is_power_point)

def save_mapping_cmp(is_power_point = False):
    dfs = []
    for filename in glob.glob("../output/cifar10classifier_resnet32_*.csv"):
        target = filename.split("_")[-1].split(".csv")[0] 
        if target in ["momentum", "bnafteraddition", "fullpreactivation", "noactivation", "relubeforeaddition", "reluonlypreactivation"]:
            df = pd.read_csv(filename)
            df["train_error"] = 1 - df["train_accuracy"]
            df["test_error"] = 1 - df["test_accuracy"]
            dfs.append(df)
    total_df = pd.concat(dfs)
    total_df["name"] = total_df["name"].str.split("_").str.get(-1).str.replace("Momentum", "Nesterov(Original Paper)")
    ax = sns.pointplot(x="epoch", y="test_error", hue="name", data=total_df, scale=0.2)
    if is_power_point:
        ax.legend(loc="lower left", markerscale=9.0, fontsize=20)
    else:
        ax.legend(markerscale=3.0)
    ax.set(ylim=(0, 0.2))
    ax.set_xticklabels([i if i % 10 == 0 else "" for i in range(200)])
    ax.set(xlabel='epoch', ylabel='error(%)')
    ax.get_figure().savefig("../figures/resnet.mapping.png")
    sns.plt.close()

def save_solvers_cmp(is_power_point = False):
    dfs = []
    for filename in glob.glob("../output/cifar10classifier_resnet32_*.csv"):
        target = filename.split("_")[-1].split(".csv")[0] 
        if target in ["adadelta", "adagrad", "adam", "momentum", "rmsprop"]:
            df = pd.read_csv(filename)
            df["train_error"] = 1 - df["train_accuracy"]
            df["test_error"] = 1 - df["test_accuracy"]
            dfs.append(df)
    total_df = pd.concat(dfs)
    total_df["name"] = total_df["name"].str.split("_").str.get(-1).str.replace("Momentum", "Nesterov(Original Paper)")
    ax = sns.pointplot(x="epoch", y="test_error", hue="name", data=total_df, scale=0.2)
    if is_power_point:
        ax.legend(loc="lower left", markerscale=9.0, fontsize=20)  
    else:
        ax.legend(loc="lower left", markerscale=3.0)
    ax.set(ylim=(0, 0.2))
    ax.set_xticklabels([i if i % 10 == 0 else "" for i in range(200)])
    ax.set(xlabel='epoch', ylabel='error(%)')
    ax.get_figure().savefig("../figures/resnet.solvers.png")
    sns.plt.close()

def save_layers_cmp(is_power_point = False):
    total_df = None
    for layer in [20, 32, 44, 56, 110]:
        df = pd.read_csv("../output/cifar10classifier_resnet%d.csv" % layer)
        df["train_error"] = 1 - df["train_accuracy"]
        df["test_error"] = 1 - df["test_accuracy"]
        df = df[df["epoch"] < 150]
        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([total_df, df])
    total_df["name"] = total_df["name"].str.split("_").str.get(-1)
    ax = sns.pointplot(x="epoch", y="test_error", hue="name", data=total_df, scale=0.2)
    if is_power_point:
        ax.legend(loc="lower left", markerscale=9.0, fontsize=20)  
    else:
        ax.legend(markerscale=3.0)
    ax.set(ylim=(0, 0.2))
    ax.set_xticklabels([i if i % 10 == 0 else "" for i in range(150)])
    ax.set(xlabel='epoch', ylabel='error(%)')
    ax.get_figure().savefig("../figures/resnet.layers.png")
    sns.plt.close()

if __name__ == "__main__":
    run()

