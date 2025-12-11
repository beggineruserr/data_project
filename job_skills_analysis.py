import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
import re



SKILLS = {
    "python": r"\bpython\b",
    "r": r"\br\b",
    "sql": r"\bsql\b",
    "java": r"\bjava\b",
    "javascript": r"\bjavascript\b",
    "typescript": r"\btypescript\b",
    "c++": r"\bc\+\+\b",
    "c#": r"\bc\#\b",
    "c": r"\bc\b",
    "golang": r"\bgolang\b",
    "rust": r"\brust\b",
    "scala": r"\bscala\b",
    "ruby": r"\bruby\b",
    "php": r"\bphp\b",
    "swift": r"\bswift\b",
    "kotlin": r"\bkotlin\b",
    "matlab": r"\bmatlab\b",
    "sas": r"\bsas\b",
    "bash": r"\bbash\b",
    "shell": r"\bshell\b",
    "perl": r"\bperl\b",
    "groovy": r"\bgroovy\b",
    "aws": r"\baws\b",
    "azure": r"\bazure\b",
    "gcp": r"\bgcp\b",
    "ec2": r"\bec2\b",
    "s3": r"\bs3\b",
    "lambda": r"\blambda\b",
    "cloudformation": r"\bcloudformation\b",
    "terraform": r"\bterraform\b",
    "kubernetes": r"\bkubernetes\b",
    "docker": r"\bdocker\b",
    "jenkins": r"\bjenkins\b",
    "ansible": r"\bansible\b",
    "linux": r"\blinux\b",
    "unix": r"\bunix\b",
    "airflow": r"\bairflow\b",
    "dbt": r"\bdbt\b",
    "hadoop": r"\bhadoop\b",
    "spark": r"\bspark\b",
    "kafka": r"\bkafka\b",
    "hive": r"\bhive\b",
    "presto": r"\bpresto\b",
    "flink": r"\bflink\b",
    "nifi": r"\bnifi\b",
    "snowflake": r"\bsnowflake\b",
    "databricks": r"\bdatabricks\b",
    "redshift": r"\bredshift\b",
    "bigquery": r"\bbigquery\b",
    "mysql": r"\bmysql\b",
    "postgresql": r"\bpostgresql\b",
    "oracle": r"\boracle\b",
    "sqlite": r"\bsqlite\b",
    "mariadb": r"\bmariadb\b",
    "mongodb": r"\bmongodb\b",
    "cassandra": r"\bcassandra\b",
    "redis": r"\bredis\b",
    "dynamodb": r"\bdynamodb\b",
    "elasticsearch": r"\belasticsearch\b",
    "neo4j": r"\bneo4j\b",
    "scikit-learn": r"\bscikit-learn\b",
    "tensorflow": r"\btensorflow\b",
    "pytorch": r"\bpytorch\b",
    "keras": r"\bkeras\b",
    "xgboost": r"\bxgboost\b",
    "lightgbm": r"\blightgbm\b",
    "nlp": r"\bnlp\b",
    "tableau": r"\btableau\b",
    "power bi": r"\bpower bi\b",
    "looker": r"\blooker\b",
    "qlik": r"\bqlik\b",
    "grafana": r"\bgrafana\b",
    "quicksight": r"\bquicksight\b",
    "metabase": r"\bmetabase\b",
    "matplotlib": r"\bmatplotlib\b",
    "seaborn": r"\bseaborn\b",
    "plotly": r"\bplotly\b",
    "d3": r"\bd3\b",
    "git": r"\bgit\b",
    "github": r"\bgithub\b",
    "gitlab": r"\bgitlab\b",
    "bitbucket": r"\bbitbucket\b",
    "jira": r"\bjira\b",
    "confluence": r"\bconfluence\b",
    "ci/cd": r"\bci/cd\b",
    "excel": r"\bexcel\b",
    "vba": r"\bvba\b",
    "powerpoint": r"\bpowerpoint\b",
    "word": r"\bword\b",
    "rest api": r"\brest api\b",
    "graphql": r"\bgraphql\b",
    "fastapi": r"\bfastapi\b",
    "flask": r"\bflask\b",
    "django": r"\bdjango\b",
    "react": r"\breact\b",
    "node": r"\bnode\b",
    "pandas": r"\bpandas\b",
    "numpy": r"\bnumpy\b",
    "scipy": r"\bscipy\b",
}

MIN_OCCURRENCES = 10




def load_data_one():
    df = pd.read_csv("data_jobs.csv")
    df = df.sample(n=20000, random_state=42)
    df["job_posted_date"] = pd.to_datetime(df["job_posted_date"], errors="coerce")
    df = df.dropna(subset=["job_posted_date"])
    df["year_month"] = df["job_posted_date"].dt.to_period("Y")
    df["job_skills"] = df["job_skills"].fillna("").astype(str)
    return df[["job_posted_date", "year_month", "job_skills"]]


def load_data_two():
    df = pd.read_csv("Engineering_Jobs_Insight_Dataset.csv")
    df = df.rename(columns={"Date Posted": "job_posted_date", "Description": "job_skills"})
    df["job_posted_date"] = pd.to_datetime(df["job_posted_date"], errors="coerce")
    df = df.dropna(subset=["job_posted_date"])
    df["year_month"] = df["job_posted_date"].dt.to_period("Y")
    df["job_skills"] = df["job_skills"].fillna("").astype(str)
    return df[["job_posted_date", "year_month", "job_skills"]]


def load_data_three():
    df = pd.read_csv("jobs.csv")
    def extract(js):
        try:
            d = json.loads(js)
            return pd.Series({
                "job_posted_date": d.get("datePosted"),
                "job_skills": d.get("description", "")
            })
        except:
            return pd.Series({"job_posted_date": None, "job_skills": ""})

    df = df["context"].apply(extract)
    df["job_posted_date"] = pd.to_datetime(df["job_posted_date"], errors="coerce")
    df = df.dropna(subset=["job_posted_date"])
    df["year_month"] = df["job_posted_date"].dt.to_period("Y")
    df["job_skills"] = df["job_skills"].fillna("").astype(str)
    return df[["job_posted_date", "year_month", "job_skills"]]




def compute_skill_counts(df):
    jobs_per_period = df["year_month"].value_counts().sort_index()
    periods = list(jobs_per_period.index)
    skill_counts = {p: Counter() for p in periods}

    for row in df.itertuples(index=False):
        text = row.job_skills.lower()
        period = row.year_month

        matched = {skill for skill, patt in SKILLS.items() if re.search(patt, text)}
        for skill in matched:
            skill_counts[period][skill] += 1

    return skill_counts, jobs_per_period, periods



def fit_trends(skill_counts, jobs_per_period, periods):
    total_counts = Counter()
    for p, counter in skill_counts.items():
        total_counts.update(counter)

    trends = []

    for skill, total in total_counts.items():
        if total < MIN_OCCURRENCES:
            continue

        y = np.array([(skill_counts[p].get(skill, 0) / jobs_per_period[p]) * 100 for p in periods])
        x = np.arange(len(periods))

        slope, intercept = np.polyfit(x, y, 1)
        mae = np.mean(np.abs((slope * x + intercept) - y))

        trends.append({
            "skill": skill,
            "slope": slope,
            "intercept": intercept,
            "mae": mae,
            "avg_frequency": y.mean(),
        })

    return pd.DataFrame(trends).sort_values("slope", ascending=False).reset_index(drop=True)



def plot_trends(trend_df):
 
    inc = trend_df[trend_df["slope"] > 0].nlargest(10, "slope")
    plt.figure(figsize=(8, 6))
    plt.barh(inc["skill"], inc["slope"], color="green")
    plt.gca().invert_yaxis()
    plt.title("Top Growing Skills (slope)")
    plt.xlabel("Slope")
    plt.savefig("increasing_skill_trends.png")
    plt.close()

    dec = trend_df[trend_df["slope"] < 0].nsmallest(10, "slope")
    plt.figure(figsize=(8, 6))
    plt.barh(dec["skill"], dec["slope"], color="red")
    plt.gca().invert_yaxis()
    plt.title("Declining Skills (slope)")
    plt.xlabel("Slope")
    plt.savefig("decreasing_skill_trends.png")
    plt.close()


def plot_top10_frequency(trend_df):
    top = trend_df.nlargest(10, "avg_frequency")
    plt.figure(figsize=(8, 6))
    plt.barh(top["skill"], top["avg_frequency"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Most Frequent Skills")
    plt.xlabel("Average Frequency (%)")
    plt.savefig("frequency_top10.png")
    plt.close()


def plot_frequency_scatter(trend_df):
    plt.figure(figsize=(12, 8))

    x = trend_df["slope"]
    y = trend_df["avg_frequency"]

    plt.scatter(x, y, alpha=0.7)

   
    for _, row in trend_df.iterrows():
        plt.text(
            row["slope"] + 0.02,      
            row["avg_frequency"] + 0.02,  
            row["skill"],
            fontsize=8,
            alpha=0.8
        )

    plt.xlabel("Slope (Trend)")
    plt.ylabel("Average Frequency (%)")
    plt.title("Skill Frequency vs Trend (Slope) with Labels")

    plt.tight_layout()
    plt.savefig("frequency_scatter.png")
    plt.close()


def plot_individual_regressions(trend_df, skill_counts, jobs_per_period, periods):
    inc = trend_df[trend_df["slope"] > 0].nlargest(10, "slope")
    dec = trend_df[trend_df["slope"] < 0].nsmallest(10, "slope")
    selected = pd.concat([inc, dec])

    x = np.arange(len(periods))
    labels = [str(p) for p in periods]

    for _, row in selected.iterrows():
        skill = row["skill"]

        y = [(skill_counts[p].get(skill, 0) / jobs_per_period[p]) * 100 for p in periods]
        pred = row["slope"] * x + row["intercept"]

        plt.figure(figsize=(8, 6))
        plt.plot(labels, y, "o-", label="Actual")
        plt.plot(labels, pred, "--", label="Regression")
        plt.xticks(rotation=45)
        plt.title(f"Trend for {skill}")
        plt.ylabel("Frequency (%)")
        plt.legend()
        safe_skill = skill.replace("/", "_").replace(" ", "_")
        plt.savefig(f"regression_{safe_skill}.png")
        plt.close()

def plot_mae_scatter(trend_df):
    plt.figure(figsize=(12, 7))

    x_values = range(len(trend_df))
    y_values = trend_df["mae"]

    plt.scatter(x_values, y_values, alpha=0.7)

    for i, row in trend_df.iterrows():
        plt.text(
            i + 0.2,
            row["mae"] + 0.1,
            row["skill"],
            fontsize=7,
            alpha=0.75
        )

    plt.title("MAE per Skill")
    plt.xlabel("Skill Index")
    plt.ylabel("MAE")

    plt.tight_layout()
    plt.savefig("mae_scatter_labeled.png")
    plt.close()


if __name__ == "__main__":
    df = pd.concat([load_data_one(), load_data_two(), load_data_three()], ignore_index=True)

    skill_counts, jobs_per_period, periods = compute_skill_counts(df)
    trend_df = fit_trends(skill_counts, jobs_per_period, periods)

    plot_trends(trend_df)  
    plot_top10_frequency(trend_df)
    plot_frequency_scatter(trend_df)
    plot_individual_regressions(trend_df, skill_counts, jobs_per_period, periods)
    plot_mae_scatter(trend_df)

    print("All plots generated successfully.")
