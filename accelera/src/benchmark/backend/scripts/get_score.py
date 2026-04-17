from utils import get_file_url
import sys
import sklearn.metrics as metrics
from accelera.src.utils.dataset_retriever import retriever
import json

true_labeled_file = sys.argv[1]
predicted_labeled_file = sys.argv[2]
target_col = sys.argv[3]
user_id = sys.argv[4]
sklearn_name = sys.argv[5]
metric_paramters = json.loads(sys.argv[6])

try:
    true_url = get_file_url(true_labeled_file)
    predicted_url = get_file_url(predicted_labeled_file)
    retriever.connect()
    true_df = retriever.retrieve_dataset(f"{user_id}_true_label", url=true_url, df=True)
    retriever.connect()
    predicted_df = retriever.retrieve_dataset(
        f"{user_id}_predicted", url=predicted_url, df=True
    )
    if target_col not in predicted_df.columns:
        print(
            json.dumps(
                {
                    "message": f"Dataset dosent have this target column {target_col} the columns exists are {predicted_df.columns}",
                    "isValid": False,
                }
            )
        )
    else:

        metric_func = getattr(metrics, sklearn_name)

        res = metric_func(
            true_df[target_col], predicted_df[target_col], **metric_paramters
        )
        print(
            json.dumps(
                {
                    "message": "Every Dataset validation correctly",
                    "isValid": True,
                    "score": res,
                }
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
