import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from warnings import warn


def clean_df(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    ACTIVITY_KEEP = ["Input", "Remove/Cut", "Nonproduction", "Replace", "Paste", "Move"]
    DOWN_EVENT_KEEP = ["q", "Space", "Backspace", "Shift", "ArrowRight", "Leftclick", "ArrowLeft",
                       ".", ",", "ArrowDown", "ArrowUp", "Enter", "CapsLock", "'", "Delete", "Unidentified",
                       "Control", "\"", "-", "?", ";", "=", "Tab", "/", "Rightclick", ":", "(", ")", "\\",
                       "ContextMenu", "End", "!", "Meta", "Alt", "[", "c", "v", "MinorKey"]
    TEXT_CHANGE_KEEP = ["q", "", "NoChange", ".", ",", "\\n", "'", "\"", "-", "?", ";", "=",
                        "/", "\\", ":", "(", "[", ")", "]", "!", "ReplaceText"]

    # remove unneeded columns
    df = df.drop(["up_time", "up_event"], axis=1)

    # clean up activity
    df.loc[~df["activity"].isin(ACTIVITY_KEEP), "activity"] = "Move"
    # clean up down_event
    df.loc[~df["down_event"].isin(DOWN_EVENT_KEEP), "down_event"] = "MinorKey"
    # clean up text_change
    df.loc[~df["text_change"].isin(TEXT_CHANGE_KEEP), "text_change"] = "ReplaceText"

    return df


def main_cli() -> dict:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="Path to input data")
    parser.add_argument("dest", help="Path to the output folder")
    parser.add_argument("-c", "--clean", action="store_true", help="Clean up data")
    return vars(parser.parse_args())


def main() -> int:
    args = main_cli()

    data = pd.read_csv(args["src"])
    if args["clean"]:
        data = clean_df(data)

    return 0


if __name__ == "__main__":
    main()
