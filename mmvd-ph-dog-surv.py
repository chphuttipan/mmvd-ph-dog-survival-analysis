# 1. import Python libraries
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

# 2. import CSV file from google drive
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/mmvd_ph_data.csv')

# 3. Data cleaning 
# 3.1 Create new table select data from first visit (column 'visit' == 1) 
df_last = df[df['visit'] == 1]

# 3.2 Delete rename columns
df_last.drop('rbc...84', axis = 1, inplace = True)
df_last = df_last.rename(columns = {'rbc...85': 'rbc'})

# 3.3 Create new column 'surv_status' by grouping feature from column 'group' and 'stage'
df_last.loc[(df_last['stage'] == 'Normal') | (df_last['stage'] == 'B1'), 'surv_status'] = 'nc_b1'
df_last.loc[(df_last['stage'] == 'B2'), 'surv_status'] = 'b2'
df_last.loc[(df_last['group'] == 'MVD') & (df_last['stage'] == 'C'), 'surv_status'] = 'cd'
df_last.loc[(df_last['group'] == 'MVDPH') & (df_last['stage'] == 'C'), 'surv_status'] = 'cd'
df_last.loc[(df_last['group'] == 'MVD') & (df_last['stage'] == 'D'), 'surv_status'] = 'cd'
df_last.loc[(df_last['group'] == 'MVDPH') & (df_last['stage'] == 'D'), 'surv_status'] = 'cd'
df_last.loc[(df_last['group'] == 'PHO') & (df_last['stage'] == 'C'), 'surv_status'] = 'ph_cd'
df_last.loc[(df_last['group'] == 'PHO') & (df_last['stage'] == 'D'), 'surv_status'] = 'ph_cd'

# 4. Create Kaplan Meier Survival Plot 
kmf = KaplanMeierFitter()
kmf.fit(df_last['max_duration'], df_last['event'])
kmf.plot()

# 5. Create Kaplan Meier Survival Plot separate by feature in column 'surv_status'
nc_b1 = df_last.query("surv_status == 'nc_b1'")
b2 = df_last.query("surv_status == 'b2'")
cd = df_last.query("surv_status == 'cd'")

kmf_nc_b1.fit(nc_b1['max_duration'], nc_b1['event'], label = "Normal-B1")
kmf_b2.fit(b2['max_duration'], b2['event'], label = "B2")
kmf_cd.fit(cd['max_duration'], cd['event'], label = "C-D")

kmf_nc_b1.plot()
kmf_b2.plot()
kmf_cd.plot()

plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.title("Kaplan Meier Survival Plot Status")

# 6. Median survival time in each group 
kmf_nc_b1.median_survival_time_
kmf_b2.median_survival_time_
kmf_cd.median_survival_time_

# 7. Log-rank test by group from 'surv_status'
time_ncb1 = nc_b1['duration']
time_b2 = b2['duration']
time_cd = cd['duration']

event_ncb1 = nc_b1['event']
event_b2 = b2['event']
event_cd = cd['event']

result_ncb1_b2 = logrank_test(time_ncb1, time_b2, event_observed_A=event_ncb1, event_observed_B=event_b2)
result_ncb1_b2.print_summary()

# 8. Cox proportional hazard analysis
# 8.1 Create a new data table by select column 'surv_status', 'event' and 'max_duration'
df_surv_status_cox = df_last[['surv_status', 'event', 'max_duration']]

# 8.2 Set a 'surv_status' column to categorical number as column 'gstatus'
df_surv_status_cox.loc[(df_surv_status_cox['surv_status'] == 'nc_b1'), 'gstatus'] = 1
df_surv_status_cox.loc[(df_surv_status_cox['surv_status'] == 'b2'), 'gstatus'] = 2
df_surv_status_cox.loc[(df_surv_status_cox['surv_status'] == 'cd'), 'gstatus'] = 3
df_surv_status_cox = df_surv_status_cox.drop('surv_status', axis = 1)

# 8.3 Cox Proportional hazard test
cph = CoxPHFitter()
cph.fit(df_surv_status_cox, "max_duration", event_col = "event")
cph.print_summary()
