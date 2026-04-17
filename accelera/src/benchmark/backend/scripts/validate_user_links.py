from accelera.src.utils.dataset_retriever import retriever
import json
import sys
from utils import get_file_url
import numpy as np


drive_file_link_test = sys.argv[1]
drive_file_link_target = sys.argv[2]
target_column_name = sys.argv[3]
user_id = sys.argv[4]

try:
    test_url = get_file_url(drive_file_link_test)
    target_file_url = get_file_url(drive_file_link_target)
    retriever.connect()
    test_df = retriever.retrieve_dataset(f"{user_id}_test", url=test_url, df=True)
    if target_column_name in test_df.columns:
        print(
            json.dumps(
                {
                    "message": f"Test Dataset has this target column {target_column_name} which it is not allowed",
                    "isValid": False,
                }
            )
        )
        retriever.close()
    else:
        retriever.connect()
        target_df = retriever.retrieve_dataset(
            f"{user_id}_target", url=target_file_url, df=True
        )
        if target_column_name not in target_df.columns:
            print(
                json.dumps(
                    {
                        "message": f"target Dataset dosent have this target column {target_column_name} the columns exists are {target_df.columns}",
                        "isValid": False,
                    }
                )
            )
        elif  not np.issubdtype(target_df[target_column_name].dtype, np.integer):
            print(
                json.dumps(
                    {"message": "Column must be number not object", "isValid": False}
                )
            )
        else:
            print(
                json.dumps(
                    {"message": "Every Dataset validation correctly", "isValid": True}
                )
            )

    sys.exit(0)

except Exception as e:
    print(json.dumps({"message": str(e), "isValid": False}))
    sys.exit(1)
finally:
    try:
        retriever.close()
    except:
        pass
