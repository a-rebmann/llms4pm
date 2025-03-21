import pandas as pd
import asyncio
import time
from baml_client.async_client import b
from dotenv import load_dotenv
import json
import os
import pickle
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

BATCH_SIZE = 50


async def main():
    res = await asyncio.gather(*(b.ExtractObjectActionPairs(
            all_unique[i:i+BATCH_SIZE]
        ) for i in range(0, len(all_unique), BATCH_SIZE)))
    return res


if __name__ == "__main__":
    corpus_df = pd.read_csv(DATA_PATH + "process_behavior_corpus.csv")
    corpus_df["unique_activities"] = corpus_df["unique_activities"].apply(lambda x: eval(x))
    # get all unique activities
    all_unique = set()

    for idx, row in corpus_df.iterrows(): 
        all_unique.update(row["unique_activities"])
    all_unique = list(all_unique)#[:100]
    print(len(all_unique), "unique activities")
    start_time = time.time()
    res = asyncio.run(main())
    print("Total time taken:", time.time() - start_time)
    all_extracted_labels = []
    for r in res:
        all_extracted_labels+= r.pairs_from_label
    
    # read pickle from extracted_labels.pkl
    with open(DATA_PATH + "extracted_labels.pkl", "rb") as f:
        all_extracted_labels = pickle.load(f)

    # create a dictionary with original_label as key and object_action_pairs as value
    extracted_labels_dict = {}
    for label in all_extracted_labels:
        extracted_labels_dict[label.original_label] = [(obj_act_pair.object, obj_act_pair.action) for obj_act_pair in label.object_action_pairs]
    
    # save the dictionary as a json file
    with open(DATA_PATH + "extracted_labels.json", "w") as f:
        json.dump(extracted_labels_dict, f)
    
    print("Done")
    