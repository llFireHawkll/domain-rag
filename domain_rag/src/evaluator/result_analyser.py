from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ResultAnalyser:
    def __init__(self, sa_score_df: pd.DataFrame, ma_score_df: pd.DataFrame) -> None:
        # Step-1: Setup internal attributes
        self.sa_score_df = sa_score_df  # Single Agent Score Dataframe
        self.ma_score_df = ma_score_df  # Multi Agent Score Dataframe

        # Step-2: Selected all the required metrics
        self.comparison_metrics = [
            "faithfulness",
            "context_precision",
            "context_recall",
            "context_entity_recall",
            "answer_relevancy",
            "answer_similarity",
            "answer_correctness",
        ]

        self.performance_metrics = ["execution_cost", "response_time"]

        self.sa_score_df = self.sa_score_df[
            self.comparison_metrics + self.performance_metrics
        ]
        self.ma_score_df = self.ma_score_df[
            self.comparison_metrics + self.performance_metrics
        ]

        # Step-3: Fillna values with `0` for all the metric
        self.sa_score_df.fillna(0, inplace=True)
        self.ma_score_df.fillna(0, inplace=True)

    def analyse_and_compare_agent_systems(self, output_path: str):
        # Step-1a: Bar Plot for Mean Comparison
        mean_df1 = self.sa_score_df[self.comparison_metrics].mean()
        mean_df2 = self.ma_score_df[self.comparison_metrics].mean()

        mean_comparison = pd.DataFrame(
            {
                "Metric": mean_df1.index,
                "Single Agent": mean_df1.values,
                "Multi Agent": mean_df2.values,
            }
        )

        # Step-1b: Bar Plot for Mean Comparison
        mean_df1 = self.sa_score_df[self.performance_metrics].mean()
        mean_df2 = self.ma_score_df[self.performance_metrics].mean()

        mean_performance = pd.DataFrame(
            {
                "Metric": mean_df1.index,
                "Single Agent": mean_df1.values,
                "Multi Agent": mean_df2.values,
            }
        )

        # Step-3: Plot Barplot and Store it at a given path
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Metric",
            y="value",
            hue="variable",
            data=pd.melt(
                mean_comparison,
                id_vars="Metric",
                value_vars=["Single Agent", "Multi Agent"],
            ),
        )
        plt.title("Mean Comparison of Metrics Between Single Agent vs Multi Agent")
        plt.xlabel("Metric")
        plt.ylabel("Mean Value")
        plt.legend(title="DataFrame")
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels
        plt.tight_layout()
        plt.savefig(output_path + "/" + "mean_comparison.png", dpi=300)  # Save plot
        plt.show()

        # Step-4: Plot Overlaid KDE Plot for Distribution Comparison
        plt.figure(figsize=(10, 6))
        for metric in self.comparison_metrics:
            sns.kdeplot(
                self.sa_score_df[metric], label=f"Single Agent - {metric}", fill=True
            )
            sns.kdeplot(
                self.ma_score_df[metric],
                label=f"Multi Agent - {metric}",
                linestyle="--",
            )
        plt.title("KDE Plot of Metrics Distributions")
        plt.xlabel("Metric Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_path + "/" + "kde_distribution_comparison.png", dpi=300
        )  # Save plot
        plt.show()

        # Step-5: Plot Scatter Plot with Regression Line for Pairwise Comparison
        plt.figure(figsize=(8, 6))
        for metric in self.comparison_metrics:
            sns.regplot(
                x=self.sa_score_df[metric], y=self.ma_score_df[metric], label=metric
            )
        plt.title("Scatter Plot with Regression for Pairwise Metric Comparison")
        plt.xlabel("Single Agent")
        plt.ylabel("Multi Agent")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path + "/" + "scatter_comparison.png", dpi=300)
        plt.show()

        # Step-6: Plot Relative Performance Bar Plot
        relative_performance = (
            (
                self.ma_score_df[self.comparison_metrics].mean()
                - self.sa_score_df[self.comparison_metrics].mean()
            )
            / self.sa_score_df[self.comparison_metrics].mean()
        ) * 100
        relative_performance_df = pd.DataFrame(
            {
                "Metric": relative_performance.index,
                "Relative Change (%)": relative_performance.values,
            }
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Metric",
            y="Relative Change (%)",
            data=relative_performance_df,
            palette="viridis",
            hue="Metric",
            legend=False,
        )
        plt.title("Relative Performance Change (%) Between Single Agent vs Multi Agent")
        plt.xlabel("Metric")
        plt.ylabel("Relative Change (%)")
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels
        plt.tight_layout()
        plt.savefig(
            output_path + "/" + "relative_performance_change.png", dpi=300
        )  # Save plot
        plt.show()

        # Step-7: Plot for Average Execution Cost
        plt.figure(figsize=(8, 6))
        sns.barplot(
            x="Metric",
            y="value",
            hue="variable",
            data=pd.melt(
                mean_performance[mean_performance["Metric"] == "execution_cost"],
                id_vars="Metric",
                value_vars=["Single Agent", "Multi Agent"],
            ),
        )
        plt.title(
            "Comparison of Average Execution Cost in USD Between Single Agent vs Multi Agent"
        )
        plt.xlabel("Metric")
        plt.ylabel("Average Value")
        plt.legend(title="DataFrame")
        plt.tight_layout()
        plt.savefig(
            output_path + "/" + "average_execution_cost_comparison.png", dpi=300
        )
        plt.show()

        # Step-8: Plot for Average Response Time
        plt.figure(figsize=(8, 6))
        sns.barplot(
            x="Metric",
            y="value",
            hue="variable",
            data=pd.melt(
                mean_performance[mean_performance["Metric"] == "response_time"],
                id_vars="Metric",
                value_vars=["Single Agent", "Multi Agent"],
            ),
        )
        plt.title(
            "Comparison of Average Response Time in Sec Between Single Agent vs Multi Agent"
        )
        plt.xlabel("Metric")
        plt.ylabel("Average Value")
        plt.legend(title="DataFrame")
        plt.tight_layout()
        plt.savefig(output_path + "/" + "average_response_time_comparison.png", dpi=300)
        plt.show()
