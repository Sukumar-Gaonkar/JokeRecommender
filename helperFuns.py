def calc_row_error(data):

    row, joke_map, cluster_map = data
    errorVal = 0.0
    for joke_col in row[1].keys():
        if row[1][joke_col] != 99.0:
            if joke_col in joke_map.columns:
                error = (row[1][joke_col] -
                         joke_map.iloc[cluster_map[cluster_map["data_index"] == int(row[0])].cluster.item()][joke_col]) ** 2
            else:
                error = abs(row[1][joke_col])
            errorVal += error

    return errorVal

##

# errorVal = 0.0
# for row in tqdm(data[:1].itertuples(), total=data.shape[0]):
#     for joke_col in row._fields[:10]:
#         if joke_col != "Index" and getattr(row, joke_col) != 99.0:
#             print(joke_col[1:] + " : " + str(joke_col[1:] in joke_map.columns))
#             if joke_col[1:] in joke_map.columns:
#                 error = (getattr(row, joke_col) -
#                          joke_map.iloc[cluster_map[cluster_map["data_index"] == getattr(row, "Index")].cluster.item()][
#                              joke_col[1:]]) ** 2
#                 # print(str(getattr(row, joke_col)) + " : " + str(joke_map.iloc[cluster_map[cluster_map["data_index"] == getattr(row, "Index")].cluster.item()][joke_col[1:]]))
#             else:
#                 print("Other")
#                 error = abs(getattr(row, joke_col))
#             # print(error)
#             errorVal += error
# errorVal = np.sqrt(errorVal)
# print(errorVal)
