import pandas as pd
import asyncio
import time
from baml_client.async_client import b
from dotenv import load_dotenv
load_dotenv()

"""
ascy client - ok
do not await return
liste von awaitables
asyncio.gather-> alle ausführen
asyncio.completed
rate limits -> nur 20 requests / sec
"""

DATA_PATH = "../data/"

APQC_LEVEL_1 = [
    "Develop and Manage Products and Services",
    "Develop and Manage Human Capital",
    "Develop Vision and Strategy",
    "Market and Sell Products and Services",
    "Deliver Physical Products",
    "Deliver Services",
    "Manage Customer Service",
    "Manage Information Technology",
    "Managee Financial Resources",
    "Acquire, Construct, and Manage Assets",
    "Manage Enterprise Risk, Compliance, Remediation, and Resiliency",
    "Manage External Relationships",
    "Develop and Manage Business Capabilities"
]

SAP_INDUSTRIES = [
    "Aerospace and defense",
    "Agribusiness",
    "Automotive",
    "Banking",
    "Chemicals",
    "Consumer products",
    "Construction and real estate",
    "Defense and security",
    "Government",
    "High tech",
    "Higher education and research",
    "Industrial manufacturing",
    "Insurance",
    "Life sciences and healthcare",
    "Media, sports, and entertainment",
    "Mill products",
    "Mining",
    "Oil, gas, and energy",
    "Professional services",
    "Retail",
    "Telecommunications",
    "Travel and transportation",
    "Utilities",
    "Wholesale distribution",
]
# https://www.sap.com/industries.html


RATE_LIMIT = 200

async def main(df):
    res = await asyncio.gather(*(b.CategorizeProcess(
            list(row["unique_activities"]),
            row["name"],
            SAP_INDUSTRIES,
            row["model_id"],
        ) for i, row in df.iterrows()))
    await asyncio.sleep(20)
    return res


if __name__ == "__main__":
    corpus_df = pd.read_csv(DATA_PATH + "process_behavior_corpus.csv")
    corpus_df["unique_activities"] = corpus_df["unique_activities"].apply(lambda x: eval(x))
    start_time = time.time()
    # split the df into batches so rate limit is not exceeded
    res = []
    for i in range(0, len(corpus_df), RATE_LIMIT):
        res += asyncio.run(main(corpus_df[i:i+RATE_LIMIT]))
    print("Total time taken:", time.time() - start_time)

    # create a dictionary with original_label as key and object_action_pairs as value
    categorized_processes = []
    for r in res:
        categorized_processes.append(
            {
                "model_id": r.process_id,
                "model_name": r.process_name,
                "unique_activities": r.unique_activities,
                "category": r.category,
            }
        )
    # create df and save as csv
    out_df = pd.DataFrame.from_records(categorized_processes)
    out_df.to_csv(DATA_PATH + "categorized_processes_by_industry.csv", index=False)
    
    print("Done")
    