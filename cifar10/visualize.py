import pandas as pd
import seaborn as sns

def run():
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
        """
        df.rename(columns={"test_accuracy": "%d_test_accuracy" % layer, "train_accuracy": "%d_train_accuracy" % layer}, inplace=True)
        if total_df is None:
            total_df = df
        else:
            total_df = pd.merge(total_df, df, on="epoch")
        """
    total_df["name"] = total_df["name"].str.split("_").str.get(-1)
    print total_df
    ax = sns.pointplot(x="epoch", y="test_error", hue="name", data=total_df, scale=0.2)
    ax.set(ylim=(0, 0.2))
    ax.set_xticklabels([i if i % 10 == 0 else "" for i in range(150)])
    ax.set(xlabel='epoch', ylabel='error(%)')
    sns.plt.show()

if __name__ == "__main__":
    run()

