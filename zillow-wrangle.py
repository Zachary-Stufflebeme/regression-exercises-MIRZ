#this function will clean the zillow data.
def clean_zillow(df):
# removes all null rows. If rows are all null they can all be deleted.
    df = df[df.isnull().sum(axis = 1) < len(df.columns)]
#if there no year built than we will set it to 0
    df.yearbuilt[df.bathroomcnt == 0.0] = df.yearbuilt[df.bathroomcnt == 0].fillna(0)
# nulls inside of bedroom count and bathroom count where yearbuilt was also null can be safely changed to 0's as you cant have a bed/bathroom if you dont have a building.
    df.calculatedfinishedsquarefeet[df.yearbuilt == 0.0] = df.calculatedfinishedsquarefeet[df.yearbuilt == 0.0].replace(np.nan,0)
#drop taxamount as it has lots of nulls and is highly correlated with taxvaluedollarcnt
    df = df.drop(columns = 'taxamount')
#drop all remaining nulls as their meaning cant be identified
    df = df.dropna()
    return df