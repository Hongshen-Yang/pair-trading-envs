import csv

def logger(model, datetime, networth, action, zscore, position, price0, price1):
    with open(f"{model}", mode='a+', newline='') as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([
            datetime, 
            networth,
            action,
            zscore,
            position,
            price0,
            price1
        ])