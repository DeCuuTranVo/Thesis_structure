import pandas as pd
from tabulate import tabulate
cross_validation_best_metrics_summary = [{'epoch': 143, 'train_loss': 0.41627747006714344, 'validation_loss': 0.44426850974559784, 'train_accuracy': 0.8188976377952756, 'validation_accuracy': 0.8245614035087719, 'train_area_under_curve': 0.8907456222933534, 'validation_area_under_curve': 0.8775}, {'epoch': 56, 'train_loss': 0.5345879923552275, 'validation_loss': 0.4830392450094223, 'train_accuracy': 0.7263779527559056, 'validation_accuracy': 0.8596491228070176, 'train_area_under_curve': 0.8126843657817109, 'validation_area_under_curve': 0.85875}, {'epoch': 133, 'train_loss': 0.43053202144801617, 'validation_loss': 0.508858859539032, 'train_accuracy': 0.8188976377952756, 'validation_accuracy': 0.8771929824561403, 'train_area_under_curve': 0.8835435887780079, 'validation_area_under_curve': 0.8525}, {'epoch': 98, 'train_loss': 0.4321763124316931, 'validation_loss': 0.5197466015815735, 'train_accuracy': 0.8090551181102362, 'validation_accuracy': 0.7543859649122807, 'train_area_under_curve': 0.8785225632335405, 'validation_area_under_curve': 0.8375}, {'epoch': 145, 'train_loss': 0.41851334273815155, 'validation_loss': 0.5484522581100464, 'train_accuracy': 0.8366141732283464, 'validation_accuracy': 0.8596491228070176, 'train_area_under_curve': 0.8896270121711818, 'validation_area_under_curve': 0.8349875930521092}, {'epoch': 78, 'train_loss': 0.4874688386917114, 'validation_loss': 0.4143090099096298, 'train_accuracy': 0.7779960707269156, 'validation_accuracy': 0.875, 'train_area_under_curve': 0.8437881109478094, 'validation_area_under_curve': 0.895483870967742}, {'epoch': 71, 'train_loss': 0.47180123813450336, 'validation_loss': 0.6733559221029282, 'train_accuracy': 0.7819253438113949, 'validation_accuracy': 0.7857142857142857, 'train_area_under_curve': 0.858000562869383, 'validation_area_under_curve': 0.8141935483870968}, {'epoch': 111, 'train_loss': 0.3955177403986454, 'validation_loss': 0.6050977557897568, 'train_accuracy': 0.8330058939096268, 'validation_accuracy': 0.7857142857142857, 'train_area_under_curve': 0.9041402170174178, 'validation_area_under_curve': 0.7870967741935484}, {'epoch': 92, 'train_loss': 0.4730840977281332, 'validation_loss': 0.5321514457464218, 'train_accuracy': 0.7956777996070727, 'validation_accuracy': 0.8035714285714286, 'train_area_under_curve': 0.8542481003158323, 'validation_area_under_curve': 0.8361290322580646}, {'epoch': 94, 'train_loss': 0.4771692790091038, 'validation_loss': 0.38783927261829376, 'train_accuracy': 0.7779960707269156, 'validation_accuracy': 0.875, 'train_area_under_curve': 0.8507145314112385, 'validation_area_under_curve': 0.9367741935483871}]
test_dict = {'test_loss': 0.5827676206827164, 'test_accuracy': 0.7142857142857143, 'test_area_under_curve': 0.7142857142857143}


'''
    In this data frame: 
        Trial 0 -> n-1 : element trials
        Trial n: means of cross_validation on validation sets
        Trial n+1: performance of best model on testset
'''
# print(len(cross_validation_best_metrics_summary))
# print(cross_validation_best_metrics_summary[0]["train_loss"])

columns = list(cross_validation_best_metrics_summary[0].keys())
columns.extend(["test_loss", "test_accuracy", "test_area_under_curve"])

df = pd.DataFrame(columns = columns)
num_rows = len(df)

for ii in range(len(cross_validation_best_metrics_summary)):
    # item_epoch = cross_validation_best_metrics_summary[ii]["epoch"]
    # item_train_loss = cross_validation_best_metrics_summary[ii]["train_loss"]
    # item_validation_loss = cross_validation_best_metrics_summary[ii]["validation_loss"]
    # item_train_accuracy = cross_validation_best_metrics_summary[ii]["train_accuracy"]
    # item_validation_accuracy = cross_validation_best_metrics_summary[ii]["validation_loss"]
    # item_train_area_under_curve = cross_validation_best_metrics_summary[ii]["train_area_under_curve"]
    # item_validation_area_under_curve = cross_validation_best_metrics_summary[ii]["validation_area_under_curve"]
    
    df = df.append(cross_validation_best_metrics_summary[ii], ignore_index=True)
    
# test_dict = {"test_loss": 0.651866, "test_accuracy": 0.706349, "test_area_under_curve": 0.785204}
# df = df.append(test_dict, ignore_index=True)

# calculate mean???
# print(type(df.mean().to_frame().T))
# print(df.mean().to_frame().T)
df = pd.concat([df,df.mean().to_frame().T], ignore_index=True)


df = df.append(test_dict, ignore_index=True)

df.to_csv('summary_result.csv', index=False)
# print(df)

# draw values table to CLI: 
# print(df.to_markdown())
print(tabulate(df, headers='keys', tablefmt='psql'))




