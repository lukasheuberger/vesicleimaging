import matplotlib.pyplot as plt
from matplotlib import pylab as pl
import os
import pandas as pd
from vesicle_imaging import format
from icecream import ic
from scipy import stats as st
from annotate_statistics import barplot_annotate_brackets


data_path = pd.ExcelFile('/Volumes/FG/Palivan/heuber0000/experimental_data/LH22-38/analysis/LH22-38_analysis.xlsx')
curr_dir = os.path.abspath(os.path.dirname(data_path))
os.chdir(curr_dir)
ic(curr_dir)

data = pd.read_excel(data_path, 'all data')
ic(data.head())
data['condition'] = pd.Categorical(data['condition'], ["empty, fresh", "empty, treated", "biofilm"])
ic(data.head())
#######################################################################
print('comparison osmolarity diff')
ic(data.head())
#define samples
group1 = data[data['condition']=='empty, fresh']
group2 = data[data['condition']=='empty, treated']
group3 = data[data['condition']=='biofilm']

#perform independent two sample t-test
#ic(st.ttest_ind(group1['diffusion coeff'], group2['diffusion coeff']))#equal_var=True)
#ic(st.ttest_ind(group1['diffusion coeff'], group2['diffusion coeff'], equal_var=True))

Ttest12, pvalue12 = st.ttest_ind(group1['diffusion constants'], group2['diffusion constants'])#, equal_var=True)
ic(Ttest12, pvalue12)

Ttest13, pvalue13 = st.ttest_ind(group1['diffusion constants'], group3['diffusion constants'], equal_var=True)
ic(Ttest13, pvalue13)

Ttest23, pvalue23 = st.ttest_ind(group2['diffusion constants'], group3['diffusion constants'], equal_var=True)
ic(Ttest23, pvalue23)

#data = data.reindex(["empty, fresh", "empty, treated", "biofilm"])
plt.figure()
format.formatLH(1.6,1.2)
data.boxplot(by='condition', grid=False)
plt.xlabel('condition')
plt.ylabel('diffusion coefficient [$\mu$m$^2$/s]')
barplot_annotate_brackets(0, 1, pvalue12, [1,2], [6.2,1])
barplot_annotate_brackets(0, 1, pvalue13, [1,3], [7,1])
barplot_annotate_brackets(0, 1, pvalue23, [2,3], [6.2,1])
plt.xticks([1, 2, 3], ['empty\nfresh', 'empty\ntreated', 'bacteria'])
plt.title('')
plt.savefig('biofilm_frap.png', dpi=150)
plt.show()






data = pd.ExcelFile('/Volumes/FG/Palivan/heuber0000/experimental_data/LH22-39/analysis/LH22-39_analysis.xlsx')
curr_dir = os.path.abspath(os.path.dirname(data))
os.chdir(curr_dir)
ic(curr_dir)

all_data = pd.read_excel(data, 'all data')
data_top_bottom = pd.read_excel(data, 'comparison top-bottom')
data_aging = pd.read_excel(data, 'aging')
data_osmo = pd.read_excel(data, 'osmolarity')


#######################################################################
print('comparison top vs. bottom')
ic(data_top_bottom.head())
#define samples
group1 = data_top_bottom[data_top_bottom['condition']=='top']
group2 = data_top_bottom[data_top_bottom['condition']=='bottom']

#perform independent two sample t-test
Ttest, pvalue = st.ttest_ind(group1['diffusion coeff'], group2['diffusion coeff'], equal_var=True)
ic(Ttest, pvalue)

plt.figure()
format.formatLH(1.7,1.2)
data_top_bottom.boxplot(by='condition', grid=False)
plt.xlabel('condition')
plt.ylabel('diffusion coefficient [$\mu$m$^2$/s]')
barplot_annotate_brackets(0, 1, pvalue, [1,2], [8,8])
plt.title('')
plt.savefig('comparison_top-bottom.png', dpi=150)
#plt.show()

#######################################################################
print('comparison fresh vs. fridge')
ic(data_aging.head())
#define samples
group1 = data_aging[data_aging['condition']=='fresh']
group2 = data_aging[data_aging['condition']=='fridge']

#perform independent two sample t-test
#ic(st.ttest_ind(group1['diffusion coeff'], group2['diffusion coeff']))#equal_var=True)
#ic(st.ttest_ind(group1['diffusion coeff'], group2['diffusion coeff'], equal_var=True))

Ttest, pvalue = st.ttest_ind(group1['diffusion coeff'], group2['diffusion coeff'], equal_var=True)
ic(Ttest, pvalue)

plt.figure()
format.formatLH(1.7,1.2)
data_aging.boxplot(by='condition', grid=False)
plt.xlabel('condition')
plt.ylabel('diffusion coefficient [$\mu$m$^2$/s]')
barplot_annotate_brackets(0, 1, pvalue, [1,2], [13,13])
plt.title('')
plt.savefig('comparison_fresh-fridge.png', dpi=150)
#plt.show()

#######################################################################
print('comparison osmolarity diff')
ic(data_osmo.head())
#define samples
group1 = data_osmo[data_osmo['condition']==-60]
group2 = data_osmo[data_osmo['condition']==19]
group3 = data_osmo[data_osmo['condition']==184]

ic(group1.head(), group2.head(), group3.head())

#perform independent two sample t-test
#ic(st.ttest_ind(group1['diffusion coeff'], group2['diffusion coeff']))#equal_var=True)
#ic(st.ttest_ind(group1['diffusion coeff'], group2['diffusion coeff'], equal_var=True))

Ttest12, pvalue12 = st.ttest_ind(group1['diffusion coeff'], group2['diffusion coeff'], equal_var=True)
ic(Ttest12, pvalue12)

Ttest13, pvalue13 = st.ttest_ind(group1['diffusion coeff'], group3['diffusion coeff'], equal_var=True)
ic(Ttest13, pvalue13)

Ttest23, pvalue23 = st.ttest_ind(group2['diffusion coeff'], group3['diffusion coeff'], equal_var=True)
ic(Ttest23, pvalue23)

plt.figure()
format.formatLH(1.7,1.2)
data_osmo.boxplot(by='condition', grid=False)
plt.xlabel('condition')
plt.ylabel('diffusion coefficient [$\mu$m$^2$/s]')
barplot_annotate_brackets(0, 1, pvalue12, [1,2], [12,12])
barplot_annotate_brackets(0, 1, pvalue13, [1,3], [14,14])
barplot_annotate_brackets(0, 1, pvalue23, [2,3], [12,12])
plt.title('')
plt.savefig('comparison_osmolarity.png', dpi=150)
#plt.show()

