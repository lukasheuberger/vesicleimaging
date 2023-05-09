dilution = []
exposure = []

for index, row in results.iterrows():
    print(row.filename)
    if row.filename.startswith('1-50'):
        dilution.append(50)
    elif row.filename.startswith('1-100'):
        dilution.append(100)
    elif row.filename.startswith('1-200'):
        dilution.append(200)

    if '_0' in row.filename:
        exposure.append(0)
    elif '_10' in row.filename:
        exposure.append(10)
    elif '_20' in row.filename:
        exposure.append(20)
    elif '_5' in row.filename:
        exposure.append(5)
    elif '_7' in row.filename:
        exposure.append(7)
    elif '_15' in row.filename:
        exposure.append(15)
    elif '_1' in row.filename:
        exposure.append(1)
    elif '_2' in row.filename:
        exposure.append(2)

results['Dilution'] = dilution
results['Exposure'] = exposure
results.to_excel('analysis_categorical.xlsx')