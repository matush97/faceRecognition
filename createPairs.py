import pandas as pd
identity = pd.read_csv("identity_CelebA.txt", delim_whitespace=True)


def create_TP_pairs(dataframe, size):
    first_image = dataframe.sample(n=size)
    dataframe = dataframe.drop(first_image.index)
    dataframe = dataframe.groupby('identity').apply(lambda x: x.sample(1)).reset_index(drop=True)

    merged = pd.merge(first_image,dataframe,how='left', on='identity')
    merged.drop(columns=['identity'], inplace=True)
    merged.to_csv('{filenameTP}', index=False)


def create_FP_pairs(dataframe,size):
    f = open("{filenameFP", "a")

    first_image = dataframe.sample(n=size)
    dataframe = dataframe.drop(first_image.index)
    for index, row in first_image.iterrows():
        paired = False
        while paired is not True:

            second_image = dataframe.sample(n=1)
            if (row['identity'] != second_image.iloc[0]['identity']):
                f.write(row['image'] +',' + second_image.iloc[0]['image']+'\n')
                paired = True

create_TP_pairs(identity,500)
create_FP_pairs(identity,500)