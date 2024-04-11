import csv

def print_info(features):
    cnt = 0
    for feature_domain in ["l", "a", "b",]:
      for param in ["mean","std","entropy"]:
          print(feature_domain + "_" + param + ": " + str(features[cnt]))
          cnt = cnt + 1
    for feature_domain in ['curvature','anisotropy','linearity','planarity','sphericity']:
      for param in ["mean","std","entropy","ggd1","ggd2","aggd1","aggd2","aggd3","aggd4", "gamma1", "gamma2"]:
          print(feature_domain + "_" + param + ": " + str(features[cnt]))
          cnt +=1
    return 1

def append_csv(csv_file_name, to_append):
    print("writing to csv file: ", csv_file_name)
    print(to_append)
    with open(csv_file_name, "a", newline= '') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(to_append)

def create_csv(csv_file_name, first_row):
    with open(csv_file_name, "w", newline= '') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(first_row)