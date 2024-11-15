import pandas as pd

def check_time_gaps(df, freq='15T'):
    """
    Controlla se ci sono buchi temporali nel DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame con indice datetime.
    - freq (str): Frequenza attesa (default: '15T').

    Returns:
    - None
    """
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing_times = expected_index.difference(df.index)
    if not missing_times.empty:
        print(f"Ci sono {len(missing_times)} timestamp mancanti nel dataset:")
        print(missing_times)
    else:
        print("Non ci sono timestamp mancanti nel dataset.")
