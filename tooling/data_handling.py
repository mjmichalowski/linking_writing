import numpy as np
import pandas as pd
from tqdm import tqdm

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from threading import Thread

# # # CONSTANTS # # #
DOWN_EVENT_KEEP = ["q", "Space", "Backspace", "Shift", "ArrowRight", "Leftclick", "ArrowLeft",
                   ".", ",", "ArrowDown", "ArrowUp", "Enter", "CapsLock", "'", "Delete", "Unidentified",
                   "Control", "\"", "-", "?", ";", "=", "Tab", "/", "Rightclick", ":", "(", ")", "\\",
                   "ContextMenu", "End", "!", "Meta", "Alt", "[", "]", "c", "v", "MinorKey"]
DOWN_EVENT_REMAP = {
        "q": ["q"],
        "Space": ["Space"],
        "Backspace": ["Backspace"],
        "Shift": ["Shift"],
        "Enter": ["Enter"],
        "CapsLock": ["CapsLock"],
        "Delete": ["Delete"],
        "MouseClick": ["Leftclick", "Rightclick"],
        "ArrowKey": ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"],
        "MajorPunctuation": [".", ",", "!", "?", "'"],
        "MinorPunctuation": ["\"", "-", ";", "=", "/", ":", "(", ")", "[", "]", "\\"],
        "MinorKey": ["MinorKey", "Unidentified", "Control", "Tab", "ContextMenu", "End",
                     "Meta", "Alt", "c", "v"]
    }
DOWN_EVENT_DICT = {
    "q": 0,
    "Space": 1,
    "Backspace": 2,
    "Shift": 3,
    "Enter": 4,
    "CapsLock": 5,
    "Delete": 6,
    "MouseClick": 7,
    "ArrowKey": 8,
    "MajorPunctuation": 9,
    "MinorPunctuation": 10,
    "MinorKey": 11
}
ACTIVITY_DICT = {
        "Input": 0,
        "Remove/Cut": 1,
        "Nonproduction": 2,
        "Replace": 3,
        "Paste": 4,
        "Move": 5
    }
TEXT_CHANGE_DICT = {
        "q": 0,
        "ReplaceText": 1,
        "": 2,
        "NoChange": 3,
        ".": 4,
        ",": 5,
        "\\n": 6,
        "'": 7,
        "\"": 8,
        "-": 9,
        "?": 10,
        ";": 11,
        "=": 12,
        "/": 13,
        "\\": 14,
        ":": 15,
        "(": 16,
        "[": 17,
        ")": 18,
        "]": 19,
        "!": 20
    }
SCALING_DICT = {
    "down_time": 1e-5,
    "action_time": 1e-2,
    "cursor_position": 1e-2,
    "word_count": 1e-2
}
# # # # # # # # # # #


def clean_df(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    # remove unneeded columns
    df = df.drop(["up_time", "up_event"], axis=1)

    print("Cleaning up data...")
    # clean up activity
    df.loc[~df["activity"].isin(ACTIVITY_DICT.keys()), "activity"] = "Move"
    # clean up down_event
    df.loc[~df["down_event"].isin(DOWN_EVENT_KEEP), "down_event"] = "MinorKey"
    # clean up text_change
    df.loc[~df["text_change"].isin(TEXT_CHANGE_DICT.keys()), "text_change"] = "ReplaceText"

    return df


def remap_down_event(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    print("Remapping down event column...")
    for remapped_event, remap_keys in tqdm(DOWN_EVENT_REMAP.items()):
        df.loc[df["down_event"].isin(remap_keys), "down_event"] = remapped_event
    return df


def scale_df(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    print("Applying scaling factors...")
    for column, scale_factor in tqdm(SCALING_DICT.items()):
        df[column] = df[column].apply(lambda x: x*scale_factor)
    return df


def vectorize_user_data(input_df: pd.DataFrame, make_orthogonal: bool = False) -> np.ndarray:
    df = input_df.copy().drop(["id"], axis=1)
    df["activity"] = df["activity"].apply(lambda x: ACTIVITY_DICT[x])
    df["down_event"] = df["down_event"].apply(lambda x: DOWN_EVENT_DICT[x])
    df["text_change"] = df["text_change"].apply(lambda x: TEXT_CHANGE_DICT[x])

    if make_orthogonal:
        output_array = np.array([])
    else:
        output_array = df.to_numpy()

    return output_array


def transform_and_save(input_df: pd.DataFrame, user_id: str, dest_path: str, label: str = "") -> None:
    df = input_df.copy()
    array = vectorize_user_data(df)
    if label:
        label_array = np.reshape(np.array([label for _ in range(8)]), (1, 8))
        array = np.concatenate((label_array, array))
    np.save(f"{dest_path}/{user_id}.npy", array)


def main_cli() -> dict:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Path to input data")
    parser.add_argument("dest", help="Path to the output folder")
    parser.add_argument("-c", "--clean", action="store_true", help="Clean up data")
    parser.add_argument("-r", "--remap", action="store_true", help="Remap down event")
    parser.add_argument("-s", "--scale", action="store_true", help="Scale data")
    parser.add_argument("-t", "--threads", default=1, type=int, help="Number of threads")
    parser.add_argument("-l", "--labels", default="", type=str, help="Path to labels")
    return vars(parser.parse_args())


def main() -> int:
    args = main_cli()
    data = pd.read_csv(args["src"])
    n_threads = args["threads"]
    dest = args["dest"]
    if args["clean"]:
        data = clean_df(data)
    if args["remap"]:
        data = remap_down_event(data)
    if args["scale"]:
        data = scale_df(data)
    if args["labels"]:
        labels = pd.read_csv(args["labels"])
        labels = {x["id"]: x["score"] for x in labels.to_dict("records")}

    user_ids = data['id'].unique().tolist()

    print("Vectorizing and saving data...")
    for i in tqdm(range(0, len(user_ids), n_threads)):
        if args["labels"]:
            threads = [Thread(target=transform_and_save, args=(data.loc[data["id"] == user_ids[j]],
                                                               user_ids[j], dest, labels[user_ids[j]]))
                       for j in range(i, i + n_threads) if j < len(user_ids)]
        else:
            threads = [Thread(target=transform_and_save, args=(data.loc[data["id"] == user_ids[j]], user_ids[j], dest))
                       for j in range(i, i+n_threads) if j < len(user_ids)]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return 0


if __name__ == "__main__":
    main()
