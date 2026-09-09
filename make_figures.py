"""Generate every manuscript figure from analysis.py outputs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

ap=argparse.ArgumentParser()
ap.add_argument('--results',type=Path,default=Path('results'))
ap.add_argument('--output',type=Path,default=Path('figures'))
a=ap.parse_args(); a.output.mkdir(exist_ok=True,parents=True)
r=a.results
d=pd.read_csv(r/'processed.csv'); d=d[d.Eligible_18_25==1].reset_index(drop=True)
inter=pd.read_csv(r/'interaction.csv'); perm=pd.read_csv(r/'permutation.csv'); dca=pd.read_csv(r/'dca.csv'); oof=pd.read_csv(r/'oof_predictions.csv')
names={'ALT_U_L':'ALT','AST_U_L':'AST','GGT_U_L':'GGT','FLI_recalc':'FLI','HSI_recalc':'HSI'}
sns.set_theme(style='whitegrid',font_scale=.9)

cols=list(names)+['BMI','WC_cm','SBP_mmHg','DBP_mmHg','FBG_mg_dL','TG_mg_dL','HDL_mg_dL']
labs=['ALT','AST','GGT','FLI','HSI','BMI','WC','SBP','DBP','FBG','TG','HDL-C']
fig,ax=plt.subplots(figsize=(9,7)); sns.heatmap(d[cols].corr(),cmap='vlag',center=0,vmin=-1,vmax=1,xticklabels=labs,yticklabels=labs,annot=True,fmt='.2f',annot_kws={'size':6},ax=ax); ax.set_title('Correlations between hepatic markers and cardiometabolic measures'); fig.tight_layout(); fig.savefig(a.output/'Figure_1_correlation.png',dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.8)); y=np.arange(len(inter)); v=inter.Interaction_OR.to_numpy(); lo=inter.Interaction_OR_L95.to_numpy(); hi=inter.Interaction_OR_U95.to_numpy(); ax.errorbar(v,y,xerr=[v-lo,hi-v],fmt='o',capsize=3); ax.axvline(1,color='black',ls='--'); ax.set_yticks(y,inter.Marker.map(names)); ax.invert_yaxis(); ax.set_xscale('log'); ax.set_xlabel('Marker-by-sex interaction OR (95% CI)'); ax.set_title('Age-adjusted interaction tests'); fig.tight_layout(); fig.savefig(a.output/'Figure_2_interactions.png',dpi=300); plt.close(fig)

def rocfig(outcome,file,title):
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,(sg,dd) in zip(axs,[('Overall',d),('Female',d[d.Sex=='F']),('Male',d[d.Sex=='M'])]):
        for m in names:
            fpr,tpr,_=roc_curve(dd[outcome],dd[m]); ax.plot(fpr,tpr,lw=1.7,label=f'{names[m]} ({roc_auc_score(dd[outcome],dd[m]):.2f})')
        ax.plot([0,1],[0,1],'k--'); ax.set(title=f'{sg} (n={len(dd)})',xlabel='1 − specificity',ylabel='Sensitivity'); ax.legend(fontsize=7,loc='lower right')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(a.output/file,dpi=300); plt.close(fig)
rocfig('MetS_recalc','Figure_3A_ROC_MetS.png','ROC performance for conventional MetS')
rocfig('MetS_star','Figure_3B_ROC_MetS_star.png','ROC performance for leakage-controlled MetS*')

pp=perm[(perm.Model_family.isin(['Age+Sex+FLI','Age+HSI (sex embedded)','Age+Sex+enzymes']))&(perm.Algorithm=='RF')].copy(); pp['Label']=pp.Model_family+' | '+pp.Feature.replace(names); pp=pp.sort_values('Permutation_AUC_drop_mean')
fig,ax=plt.subplots(figsize=(8,5)); ax.barh(pp.Label,pp.Permutation_AUC_drop_mean,xerr=pp.Permutation_AUC_drop_SD,color='#4c9f70'); ax.axvline(0,color='black',lw=.8); ax.set(xlabel='Mean decrease in outer-fold AUC after permutation',title='Cross-validated permutation importance'); fig.tight_layout(); fig.savefig(a.output/'Figure_4_permutation_importance.png',dpi=300); plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10,4.2)); yy=oof.MetS_star.to_numpy(); pcs={'Age+Sex base__LR':'Age + sex','Age+Sex+FLI__LR':'Age + sex + FLI','Enzymes only__LR':'Enzymes only','HSI only__LR':'HSI only'}
for c,l in pcs.items():
    obs,pr=calibration_curve(yy,oof[c],n_bins=6,strategy='quantile'); axs[0].plot(pr,obs,marker='o',label=l)
axs[0].plot([0,1],[0,1],'k--'); axs[0].set(xlabel='Mean predicted probability',ylabel='Observed proportion',title='Out-of-fold calibration'); axs[0].legend(fontsize=7)
for fam,g in dca.groupby('Model_family'): axs[1].plot(g.Threshold,g.Net_benefit,label=fam)
g=dca[dca.Model_family==dca.Model_family.iloc[0]]; axs[1].plot(g.Threshold,g.Treat_all,'k--',label='Treat all'); axs[1].axhline(0,color='black',label='Treat none'); axs[1].set(xlabel='Threshold probability',ylabel='Net benefit',title='Decision-curve analysis'); axs[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(a.output/'Figure_5_calibration_DCA.png',dpi=300); plt.close(fig)
print(f'Figures written to {a.output.resolve()}')
