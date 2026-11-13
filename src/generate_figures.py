from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.special import expit
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Rectangle, FancyBboxPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpecFromSubplotSpec
from .cv_and_sampling import coupling
from .load_and_qc import STATES
def save(fig, number):
    for extension in ['pdf', 'png']:
        fig.savefig(OUT / f'Figure_{number}.{extension}', dpi=600)
    plt.close(fig)

def panel(ax,letter,title):
 ax.set_title(title,loc='left',fontweight='bold',pad=13)
 ax.text(-.045/ax.get_position().width,1.045,letter,transform=ax.transAxes,fontsize=18,fontweight='bold',ha='left',va='bottom')
def ticks(ax,axis='x'):
 loc=np.log2([.125,.5,2,8,32]);lab=['0.125','0.5','2','8','32']
 if axis=='x':ax.set_xticks(loc,lab)
 else:ax.set_yticks(loc,lab)
def clean(ax):
 ax.spines[['top','right']].set_visible(False)
 ax.tick_params(direction='out',length=3,width=.6)
def hc3_band(x,y,xx):
 X=np.c_[np.ones(len(x)),x];b=np.linalg.lstsq(X,y,rcond=None)[0]
 u=y-X@b;bread=np.linalg.pinv(X.T@X);h=np.sum((X@bread)*X,axis=1)
 scores=X*(u/np.maximum(1-h,1e-8))[:,None];cov=bread@scores.T@scores@bread
 Q=np.c_[np.ones(len(xx)),xx];pred=Q@b;se=np.sqrt(np.einsum('ij,jk,ik->i',Q,cov,Q));c=stats.t.ppf(.975,len(x)-2)
 return pred,pred-c*se,pred+c*se
def isoratios(ax,xlo=-4.3,xhi=4.8,adjust=False,labels=False):
 xx=np.linspace(xlo,xhi,100)
 for r in [.5,1,2,4,8]:
  yy=(1-beta)*xx+np.log2(r)-alpha if adjust else xx+np.log2(r)
  ax.plot(xx,yy,'--',color='#AAB5BD',lw=.8,alpha=.65,zorder=0)
  if labels:
   j=50;ax.text(xx[j],yy[j],f' R={r:g}',fontsize=8,color='#8A959E',rotation=35)
def kde_contours(ax,q,x='x1',y='y1',color='#0072B2'):
 values=q[[x,y]].dropna().to_numpy().T
 if values.shape[1]<8:return
 xx=np.linspace(-4.2,4.8,85); yy=np.linspace(-3.8,5.4,85) if y=='y1' else np.linspace(-2.4,3.1,85)
 X,Y=np.meshgrid(xx,yy);Z=stats.gaussian_kde(values)(np.vstack([X.ravel(),Y.ravel()])).reshape(X.shape)
 lev=np.quantile(Z[Z>np.max(Z)*.04],[.6,.8,.93])
 ax.contour(X,Y,Z,levels=lev,colors=[color],linewidths=[.6,.9,1.1],alpha=.7)
def joint(fig,spec,letter,title,y='y1'):
 gs=spec.subgridspec(5,5,hspace=.08,wspace=.08,height_ratios=[.18,1,1,1,1],width_ratios=[1,1,1,1,.2])
 ax=fig.add_subplot(gs[1:,:4]);top=fig.add_subplot(gs[0,:4],sharex=ax);right=fig.add_subplot(gs[1:,4],sharey=ax)
 xx=np.linspace(-4.4,4.8,220);yy=np.linspace(-3.8,5.4,220) if y=='y1' else np.linspace(-2.5,3.1,220)
 for dead,c in [(0,SURV),(1,DEAD)]:
  q=D[D.death.eq(dead)]
  ax.scatter(q.x1,q[y],s=15,color=c,alpha=.43,lw=0,zorder=2)
  kde_contours(ax,q,y=y,color=c)
  zx=stats.gaussian_kde(q.x1)(xx);zy=stats.gaussian_kde(q[y])(yy)
  top.fill_between(xx,0,zx,color=c,alpha=.24);top.plot(xx,zx,color=c,lw=1)
  right.fill_betweenx(yy,0,zy,color=c,alpha=.24);right.plot(zy,yy,color=c,lw=1)
 top.axis('off');right.axis('off');panel(top,letter,title)
 ax.set_xlim(-4.3,4.8);ax.set_ylim((-3.8,5.2) if y=='y1' else (-2.4,3.1));ticks(ax)
 if y=='y1':ticks(ax,'y')
 ax.set_xlabel('Basal perfusion B (PI; log₂ scale)')
 ax.set_ylabel('Sampled post-occlusion peak P' if y=='y1' else 'Baseline-adjusted reactivity E (log₂)')
 return ax
def illustrative_case():
 q=D[D.state1.eq('LowB_AboveExpected')&D.death.eq(1)].copy()
 z=q[['x1','E1']];target=z.median();scale=z.std()
 q['distance']=(((z-target)/scale)**2).sum(axis=1)
 return q.sort_values(['distance','patient_key']).iloc[0]
def fig1():
 fig=plt.figure(figsize=(12.8,13.1))
 gs=fig.add_gridspec(3,2,height_ratios=[.85,1.2,1],left=.09,right=.96,bottom=.07,top=.95,hspace=.43,wspace=.34)
 sub=gs[0,0].subgridspec(2,1,height_ratios=[2.3,1],hspace=.38)
 ax=fig.add_subplot(sub[0]);panel(ax,'A','Common gain cancels within PI')
 t=np.linspace(0,3.5,1100);phase=t%1
 pulse=np.exp(-((phase-.23)/.095)**2)+.21*np.exp(-((phase-.53)/.12)**2)
 base=1.0;wave=base+.12*pulse
 ax.plot(t,wave,color=SURV,lw=1.7,label='Gain 1×')
 ax.plot(t,1.65*wave,color=DEAD,lw=1.7,label='Gain 1.65×')
 for dc,c in [(1,SURV),(1.65,DEAD)]:ax.axhline(dc,color=c,lw=.6,ls=':')
 ax.annotate('',xy=(2.23,1.12),xytext=(2.23,1),arrowprops=dict(arrowstyle='<->',color=SURV,lw=1))
 ax.text(2.34,1.075,'AC',fontsize=9,color=SURV)
 ax.annotate('',xy=(3.35,1),xytext=(3.35,0),arrowprops=dict(arrowstyle='<->',color='#788791',lw=.9))
 ax.text(3.17,.52,'DC',fontsize=9,rotation=90,color=MUTED)
 ax.set(xlim=(0,3.6),ylim=(0,1.98),ylabel='Optical intensity (a.u.)');ax.set_xticks([])
 ax.legend(frameon=False,ncol=2,loc='center left',bbox_to_anchor=(.01,.64),fontsize=9)
 ax.text(.02,.06,'Synthetic PPG',transform=ax.transAxes,fontsize=9,color=MUTED)
 ax2=fig.add_subplot(sub[1]);ax2.plot(t,100*(wave/base-1),color=SURV,lw=1.8)
 ax2.plot(t,100*(1.65*wave/(1.65*base)-1),color=DEAD,lw=1,ls='--')
 ax2.set(xlim=(0,3.6),ylim=(-.5,13.5),xlabel='Time (s)',ylabel='AC/DC (%)');ax2.set_yticks([0,12])
 ax=fig.add_subplot(gs[0,1]);panel(ax,'B','Baseline changes the relative increment')
 ax.set_xlim(-.55,1.55);ax.set_ylim(-.15,1.55);ax.axis('off')
 for j,(b,p,r0,r1,c,label) in enumerate([(.2,.7,.11,.19,DEAD,'Low baseline'),(8,9,.28,.30,SURV,'High baseline')]):
  cx=j;cy=1.08
  ax.add_patch(Circle((cx,cy),.33,facecolor='#F0F3F5',edgecolor='#AEBAC3',lw=1))
  ax.add_patch(Circle((cx,cy),r1,facecolor=c,alpha=.18,edgecolor=c,lw=1.4,linestyle='--'))
  ax.add_patch(Circle((cx,cy),r0,facecolor=c,alpha=.7,edgecolor=c,lw=1.4))
  ax.text(cx,1.49,label,ha='center',fontweight='bold',fontsize=11)
  ax.text(cx,.63,'Small basal lumen' if j==0 else 'Little additional dilation',ha='center',fontsize=9)
  ax.bar([cx-.12,cx+.12],[b/25,p/25],width=.17,bottom=-.02,color=['#A8B8C4',c],zorder=3)
  ax.text(cx,-.13,f'B={b:g}  P={p:g}  R={p/b:g}',ha='center',fontsize=9)
 ax.text(.5,.42,'Illustrative vessel states and PI values',ha='center',fontsize=9,color=MUTED)
 ax=joint(fig,gs[1,0],'C','The observed baseline–peak atlas','y1');isoratios(ax)
 xx=np.linspace(-4.3,4.8,200);m,lo,hi=hc3_band(D.x1.to_numpy(),D.y1.to_numpy(),xx)
 ax.fill_between(xx,lo,hi,color=INK,alpha=.13);ax.plot(xx,m,color=INK,lw=1.7)
 p=D[D.patient_key.eq('D152')].iloc[0];case=illustrative_case()
 ax.scatter([p.x1,p.x2],[p.y1,p.y2],color='#7B4CA0',s=55,zorder=6,edgecolor='white',lw=.6)
 ax.annotate('',(p.x2,p.y2),(p.x1,p.y1),arrowprops=dict(arrowstyle='-|>',lw=1.7,color='#7B4CA0'))
 ax.annotate('D152: B↑, P↑, R↓',xy=(p.x2,p.y2),xytext=(-.45,4.8),fontsize=9,color='#70418E',
  bbox=dict(boxstyle='round,pad=.25',fc='white',ec='#BDA6CD',alpha=.95),arrowprops=dict(arrowstyle='-',color='#9B7FB2',lw=.8))
 ax.scatter(case.x1,case.y1,marker='X',s=65,c=DEAD,edgecolor='white',lw=.6,zorder=7)
 ax.annotate(f'{case.patient_key}: R={case.R1:.2f}; death',xy=(case.x1,case.y1),xytext=(-3.9,3.05),fontsize=9,
  bbox=dict(boxstyle='round,pad=.25',fc='white',ec='#E5B296',alpha=.95),arrowprops=dict(arrowstyle='-',color=DEAD,lw=.8))
 ax=joint(fig,gs[1,1],'D','A fixed reference separates two coordinates','E1');isoratios(ax,adjust=True)
 ax.axvline(xc,color=MUTED,ls=':',lw=.9);ax.axhline(0,color=MUTED,ls=':',lw=.9)
 ax.scatter([p.x1,p.x2],[p.E1,p.E2],color='#7B4CA0',s=50,zorder=5,edgecolor='white',lw=.6)
 ax.annotate('',(p.x2,p.E2),(p.x1,p.E1),arrowprops=dict(arrowstyle='-|>',lw=1.8,color='#7B4CA0'))
 ax.annotate('D152',xy=(p.x2,p.E2),xytext=(3.35,1.65),fontsize=9,color='#70418E',arrowprops=dict(arrowstyle='-',color='#9B7FB2',lw=.8))
 ax.scatter(case.x1,case.E1,marker='X',s=65,c=DEAD,edgecolor='white',lw=.6,zorder=7)
 ax.legend([Line2D([],[],marker='o',color=SURV,lw=1),Line2D([],[],marker='o',color=DEAD,lw=1)], ['Survivors (n=136)','Nonsurvivors (n=90)'],frameon=False,loc='upper right',fontsize=9)
 ax=fig.add_subplot(gs[2,:]);panel(ax,'E','The same map carries bedside physiological context')
 q=D.dropna(subset=['crt_sec_d1','lactate_d1']).copy()
 kde_contours(ax,D,y='E1',color='#B7C2CA')
 ax.axvline(xc,color=MUTED,ls=':',lw=.9);ax.axhline(0,color=MUTED,ls=':',lw=.9)
 norm=colors.Normalize(1,8);cmap=plt.colormaps['viridis']
 for sh,marker in [(0,'o'),(1,'D')]:
  z=q[q.shock.eq(sh)];sc=ax.scatter(z.x1,z.E1,s=18+z.lactate_d1.clip(upper=8)*17, c=z.crt_sec_d1,cmap=cmap,norm=norm,marker=marker,alpha=.8,edgecolor=INK if sh else 'white',lw=.55,zorder=3)
 z=q[q.shock.isna()]
 if len(z):ax.scatter(z.x1,z.E1,s=18+z.lactate_d1.clip(upper=8)*17,c=z.crt_sec_d1,cmap=cmap,norm=norm,marker='^',edgecolor=INK,lw=.5)
 ax.set(xlim=(-4.3,4.8),ylim=(-2.4,3.1),xlabel='Basal perfusion B (PI; log₂ scale)',ylabel='Baseline-adjusted reactivity E (log₂)');ticks(ax)
 cb=fig.colorbar(sc,ax=ax,pad=.018,fraction=.022);cb.set_label('CRT (s; capped at 8)')
 handles=[Line2D([],[],marker='o',ls='',markerfacecolor='#8195A6',markeredgecolor='white',markersize=7,label='No shock'),
  Line2D([],[],marker='D',ls='',markerfacecolor='#8195A6',markeredgecolor=INK,markersize=7,label='Septic shock')]
 ax.legend(handles=handles,loc='upper right',frameon=True,facecolor='white',edgecolor='none',fontsize=9)
 for val,x in [(2,2.6),(4,3.4),(8,4.2)]:
  ax.scatter(x,-1.85,s=18+val*17,facecolor='none',edgecolor=MUTED,lw=.8);ax.text(x,-2.22,str(val),ha='center',fontsize=8)
 ax.text(3.4,-1.25,'Lactate (mmol/L)',ha='center',fontsize=9,color=MUTED)
 save(fig,1)
def fig2():
 fig,axs=plt.subplots(3,2,figsize=(12.8,12.2));fig.subplots_adjust(left=.09,right=.96,top=.95,bottom=.075,wspace=.39,hspace=.47)
 for day,ax in zip([1,2],axs[0]):
  q=D.dropna(subset=[f'x{day}',f'y{day}']);x=q[f'x{day}'].to_numpy();y=q[f'y{day}'].to_numpy();f=AL[AL.day.eq(day)].iloc[0]
  panel(ax,chr(64+day),f'Day {day}: sublinear scaling, n={len(q)}')
  xlo=-7 if day==2 else -4.3; ylo=-4.5 if day==2 else -3.8
  isoratios(ax,xlo=xlo,labels=False)
  ax.scatter(x,y,s=19,color=SURV if day==1 else '#009E73',alpha=.48,edgecolor='white',lw=.2)
  xx=np.linspace(xlo,4.8,160);m,lo,hi=hc3_band(x,y,xx)
  ax.fill_between(xx,lo,hi,color=SURV if day==1 else '#009E73',alpha=.2);ax.plot(xx,m,color=SURV if day==1 else '#009E73',lw=2)
  ax.text(.03,.96,f'β = {f.beta:.3f}  (95% CI {f.lower95:.3f}–{f.upper95:.3f})',transform=ax.transAxes,va='top',fontsize=10, bbox=dict(fc='white',ec='none',alpha=.9))
  ax.text(.98,.02,'Dashed: R = 0.5, 1, 2, 4, 8',transform=ax.transAxes,ha='right',fontsize=9,color=MUTED)
  ax.set(xlim=(xlo,4.8),ylim=(ylo,5.4),xlabel='Basal perfusion B (PI; log₂ scale)',ylabel='Sampled peak P (PI; log₂ scale)');ticks(ax);ticks(ax,'y')
  if day==2:
   ax.set_xticks(np.log2([.01,.125,.5,2,8,32]),['0.01','0.125','0.5','2','8','32'])
   ax.set_yticks(np.log2([.0625,.25,1,4,16]),['0.0625','0.25','1','4','16'])
 ax=axs[1,0];panel(ax,'C','Slope estimates across error assumptions')
 boots=T['allometry_bootstrap']
 ys=[4.6,3.8,3.0,1.7,.9,.1];labs=[]
 for day,yy,col in [(1,ys[:3],SURV),(2,ys[3:],'#009E73')]:
  q=AL[AL.day.eq(day)].iloc[0];b=boots[boots.day.eq(day)].iloc[0]
  for y0,est,lo,hi,mk in [(yy[0],q.beta,q.lower95,q.upper95,'o'),(yy[1],q.beta,b.bootstrap_lower95,b.bootstrap_upper95,'s')]:
   ax.errorbar(est,y0,xerr=[[est-lo],[hi-est]],fmt=mk,color=col,lw=1.5,capsize=3,ms=6)
  vals=DM[DM.day.eq(day)].beta_deming;ax.scatter(vals,[yy[2]]*3,color=col,marker='x',s=42)
  labs += [f'D{day} · OLS HC3',f'D{day} · Bootstrap',f'D{day} · Deming']
 ax.axvline(1,color=DEAD,ls='--',lw=1.4);ax.set(xlim=(.45,1.07),ylim=(-.65,5.3),xlabel='Scaling exponent β');ax.set_yticks(ys,labs,fontsize=10)
 ax.text(.98,.025,'Deming error ratios: 0.25, 1, 4',transform=ax.transAxes,ha='right',fontsize=9,color=MUTED,bbox=dict(fc='white',ec='none',alpha=.95,pad=1.5))
 ax=axs[1,1];panel(ax,'D','The rebound ratio remains baseline-dependent')
 bins=pd.qcut(D.x1,8,duplicates='drop');g=D.groupby(bins,observed=True)
 bx=g.x1.median();q10=g.r1.quantile(.1);q25=g.r1.quantile(.25);q75=g.r1.quantile(.75);q90=g.r1.quantile(.9)
 ax.fill_between(bx,q10,q90,color=SURV,alpha=.09,label='10th–90th percentile');ax.fill_between(bx,q25,q75,color=SURV,alpha=.15,label='25th–75th percentile')
 ax.scatter(D.x1,D.r1,s=17,color=SURV,alpha=.42,lw=0)
 xx=np.linspace(-4.3,4.8,150);ax.plot(xx,alpha+(beta-1)*xx,color=DEAD,lw=2)
 ax.set(xlim=(-4.3,4.8),xlabel='Basal perfusion B (PI; log₂ scale)',ylabel='Conventional rebound ratio R (log₂)');ticks(ax)
 ax.legend(frameon=False,fontsize=9,loc='upper right');ax.text(.04,.06,f'Slope = {beta-1:.3f}',transform=ax.transAxes,color=DEAD,fontsize=11)
 ax=axs[2,0];panel(ax,'E','Continuous adjusted mortality landscape')
 co=A[A.model.eq('mortality_minimal_BE')].set_index('term').effect.apply(np.log)
 xx=np.linspace(D.x1.min(),D.x1.max(),110);yy=np.linspace(D.y1.min(),D.y1.max(),110);X,Y=np.meshgrid(xx,yy);E=Y-alpha-beta*X
 risk=expit(co['intercept']+co['age']*D.age.median()+co['sofa_d1']*D.sofa_d1.median()+co['x1']*X+co['E1']*E)
 inside=Delaunay(D[['x1','y1']]).find_simplex(np.c_[X.ravel(),Y.ravel()]).reshape(X.shape)>=0
 risk=np.ma.masked_where(~inside,risk)
 im=ax.contourf(X,Y,risk,levels=np.linspace(.1,.9,17),cmap='YlOrRd',extend='both')
 cc=ax.contour(X,Y,risk,levels=[.2,.3,.4,.5,.6,.7,.8],colors='white',linewidths=.7,alpha=.9);ax.clabel(cc,fmt='%.1f',fontsize=8)
 ax.scatter(D.x1,D.y1,s=10,facecolor='none',edgecolor=INK,lw=.35,alpha=.48)
 ax.set(xlim=(-4.3,4.8),ylim=(-3.8,5.4),xlabel='Basal perfusion B (PI; log₂ scale)',ylabel='Sampled peak P (PI; log₂ scale)');ticks(ax);ticks(ax,'y')
 cbar=fig.colorbar(im,ax=ax,pad=.025,fraction=.045,ticks=[.2,.4,.6,.8]);cbar.set_label('Adjusted mortality probability',fontsize=10)
 ax.text(.03,.025,'Age 63 years; SOFA 8',transform=ax.transAxes,fontsize=9,bbox=dict(fc='white',ec='none',alpha=.8))
 ax=axs[2,1];panel(ax,'F','Retaining baseline attenuates the ratio signal')
 for i,(name,col,label) in enumerate([('mortality_unadjusted_r1',DEAD,'R alone'),('mortality_unadjusted_x1_r1',SURV,'R with basal B')]):
  q=A[A.model.eq(name)&A.term.eq('r1')].iloc[0]
  ax.errorbar(q.effect,1-i,xerr=[[q.effect-q.lower95],[q.upper95-q.effect]],fmt='o',color=col,lw=2,capsize=5,ms=8)
  ax.text(.7,1-i+.20,label,fontsize=11,color=col)
  ax.text(q.effect,1-i-.21,f'{q.effect:.2f} ({q.lower95:.2f}–{q.upper95:.2f})',ha='center',fontsize=10,fontweight='bold')
 ax.axvline(1,color=MUTED,ls='--',lw=1);ax.set_xscale('log');ax.set_xlim(.65,2.8);ax.set_ylim(-.55,1.6);ax.set_yticks([])
 ax.set_xticks([.75,1,1.5,2,2.5],['0.75','1','1.5','2','2.5']);ax.minorticks_off();ax.set_xlabel('Mortality OR per doubling of R (95% CI)')
 ax.spines['left'].set_visible(False)
 save(fig,2)
def raincloud(ax,field,title,letter):
 panel(ax,letter,title)
 for i,st in enumerate(ORDER):
  y=D.loc[D.state1.eq(st),field].dropna().to_numpy();x=i
  if np.std(y)>0:
   lo,hi=np.min(y),np.max(y);grid=np.linspace(lo,hi,180);kde=stats.gaussian_kde(y)(grid);width=.32*kde/kde.max()
   ax.fill_betweenx(grid,x-.08-width,x-.08,color=CM[st],alpha=.55,lw=0)
  ax.boxplot([y],positions=[x],widths=.10,showfliers=False,patch_artist=True,
   boxprops=dict(facecolor='white',edgecolor=INK,linewidth=.8),medianprops=dict(color=INK,linewidth=1.2),
   whiskerprops=dict(color=INK,linewidth=.8),capprops=dict(color=INK,linewidth=.8))
  jitter=RNG.uniform(.10,.28,len(y));ax.scatter(x+jitter,y,s=8,color=CM[st],alpha=.55,lw=0)
  ax.text(x,-.24,f'n={len(y)}',transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color=MUTED)
 ax.set_xticks(range(4),['Low B\nE<0','Low B\nE≥0','High B\nE<0','High B\nE≥0']);ax.set_xlim(-.55,3.5)
 ax.set_ylim(bottom=0)
def publication_forest(fig,spec,letter,title,items):
 sg=spec.subgridspec(1,3,width_ratios=[1.3,1.5,1.35],wspace=.04)
 left=fig.add_subplot(sg[0]);ax=fig.add_subplot(sg[1]);right=fig.add_subplot(sg[2])
 panel(left,letter,title);n=len(items)
 for a in [left,ax,right]:a.set_ylim(-.65,n-.35)
 for a in [left,right]:a.set_xlim(0,1);a.axis('off')
 for i,(model,term,label,c) in enumerate(items):
  y=n-1-i;q=A[A.model.eq(model)&A.term.eq(term)].iloc[0]
  left.text(0,y,label,va='center',fontsize=10)
  right.text(0,y,f'{q.effect:.2f} ({q.lower95:.2f}–{q.upper95:.2f})',va='center',fontsize=10)
  ax.errorbar(q.effect,y,xerr=[[q.effect-q.lower95],[q.upper95-q.effect]],fmt='o',color=c,lw=1.7,capsize=3,ms=6)
 ax.axvline(1,color=MUTED,ls='--',lw=.9);ax.set_yticks([]);ax.set_xscale('log');ax.set_xlim(.45,4.8)
 ax.set_xticks([.5,1,2,4],['0.5','1','2','4']);ax.minorticks_off();ax.set_xlabel('Odds ratio',fontsize=10);ax.spines['left'].set_visible(False)
 right.text(0,n-.4,'OR (95% CI)',fontweight='bold',fontsize=10,va='bottom')
def fig3():
 fig=plt.figure(figsize=(13.8,12.6));gs=fig.add_gridspec(3,6,left=.105,right=.97,top=.945,bottom=.07,height_ratios=[1.27,1,.82],hspace=.55,wspace=.60)
 q=D.assign(order=D.state1.map({s:i for i,s in enumerate(ORDER)})).sort_values(['order','x1','patient_key'])
 sg=gs[0,:].subgridspec(12,1,hspace=.08,height_ratios=[.7,.7,.7,1,1,1,1,1,1,1,.7,.37])
 groups=[('Recorded death',q.death,[SURV,DEAD]),('Septic shock',q.shock,['#DCE5EC','#465B6C']),('Display region',q.order,COLS)]
 axs=[]
 for i,(name,val,palette) in enumerate(groups):
  ax=fig.add_subplot(sg[i]);arr=np.asarray(val,float)[None,:];cm=colors.ListedColormap(palette);cm.set_bad('#D5D7D8')
  ax.imshow(arr,aspect='auto',interpolation='nearest',cmap=cm,vmin=0,vmax=len(palette)-1)
  ax.set_yticks([0],[name],fontsize=10);ax.tick_params(left=False,bottom=False,labelbottom=False);ax.spines[:].set_visible(False);axs.append(ax)
 features=[('x1','log₂ basal B'),('y1','log₂ sampled P'),('r1','log₂ ratio R'),('E1','Reactivity E'),('crt_sec_d1','CRT'),('loglac1','log₂ lactate'),('sofa_d1','SOFA')]
 diver=plt.colormaps['RdBu_r'].copy();diver.set_bad('#D5D7D8')
 for i,(key,label) in enumerate(features,3):
  ax=fig.add_subplot(sg[i]);vals=q[key];z=(vals-vals.mean())/vals.std()
  im=ax.imshow(z.to_numpy()[None,:],aspect='auto',interpolation='nearest',cmap=diver,vmin=-2.5,vmax=2.5)
  ax.set_yticks([0],[label],fontsize=10);ax.tick_params(left=False,bottom=False,labelbottom=False);ax.spines[:].set_visible(False);axs.append(ax)
 ax=fig.add_subplot(sg[10]);ne=np.where(q.norepi1.isna(),np.nan,(q.norepi1>0).astype(float));cm=colors.ListedColormap(['#DCE5EC','#465B6C']);cm.set_bad('#D5D7D8')
 ax.imshow(ne[None,:],aspect='auto',cmap=cm,vmin=0,vmax=1,interpolation='nearest');ax.set_yticks([0],['NE use (recorded)'],fontsize=10);ax.tick_params(left=False,bottom=False,labelbottom=False);ax.spines[:].set_visible(False);axs.append(ax)
 panel(axs[0],'A','Patient-level physiological landscape · 226 aligned observations')
 bounds=np.r_[0,np.cumsum([sum(q.state1.eq(s)) for s in ORDER])]
 for aa in axs:
  for b in bounds[1:-1]:aa.axvline(b-.5,color='white',lw=2)
 labelax=fig.add_subplot(sg[11]);labelax.set_xlim(-.5,225.5);labelax.set_ylim(0,1);labelax.axis('off')
 for j in range(4):labelax.text((bounds[j]+bounds[j+1])/2-.5,.4,LABELS[j].replace('\n',' / '),ha='center',fontsize=10,color=COLS[j],fontweight='bold')
 cbax=fig.add_axes([.79,.632,.15,.012]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[-2.5,0,2.5]);cb.ax.tick_params(labelsize=8);cb.set_label('Within-row z score (clipped)',fontsize=8,labelpad=1)
 fig.text(.105,.605,'Outcome: ',fontsize=9);fig.text(.159,.605,'● survivor',fontsize=9,color=SURV);fig.text(.228,.605,'● death',fontsize=9,color=DEAD)
 fig.text(.295,.605,'■ missing',fontsize=9,color='#A6AAAD');fig.text(.375,.605,'Shock / NE: pale = no; dark = yes',fontsize=9,color=MUTED)
 ax=fig.add_subplot(gs[1,:2]);panel(ax,'B','Mortality across display regions')
 for i,st in enumerate(ORDER):
  z=ST.loc[st];ax.bar(i,z.mortality*100,width=.62,color=CM[st],alpha=.9)
  for width,alp in [(7,.08),(4,.16),(1.2,1)]:
   ax.plot([i,i],[z.mortality_lower95*100,z.mortality_upper95*100],color=CM[st] if width>2 else INK,lw=width,alpha=alp)
  ax.plot([i-.09,i+.09],[z.mortality_lower95*100]*2,color=INK,lw=1);ax.plot([i-.09,i+.09],[z.mortality_upper95*100]*2,color=INK,lw=1)
  ax.text(i,z.mortality_upper95*100+3,f'{int(z.deaths)}/{int(z.n)}',ha='center',fontsize=10)
  ax.text(i,2,f'{100*z.mortality:.1f}%',ha='center',fontsize=9,color='white',fontweight='bold')
 ax.set_xticks(range(4),['Low B\nE<0','Low B\nE≥0','High B\nE<0','High B\nE≥0']);ax.set(ylabel='Recorded mortality (%)',ylim=(0,82))
 ax=fig.add_subplot(gs[1,2:4]);raincloud(ax,'crt_sec_d1','Capillary refill time','C');ax.set_ylabel('CRT (s)')
 ax=fig.add_subplot(gs[1,4:]);raincloud(ax,'lactate_d1','Circulating lactate','D');ax.set_ylabel('Lactate (mmol/L)')
 publication_forest(fig,gs[2,:3],'E','Full-cohort adjusted associations',[
  ('mortality_minimal_BE','x1','Basal B · minimal',SURV),('mortality_minimal_BE','E1','Reactivity E · minimal','#009E73'),
  ('mortality_extended_BE','x1','Basal B · extended',SURV),('mortality_extended_BE','E1','Reactivity E · extended','#009E73')])
 publication_forest(fig,gs[2,3:],'F','Septic-shock subgroup',[ ('mortality_shock_1','x1','Basal B',SURV),('mortality_shock_1','E1','Reactivity E','#009E73')])
 fig.text(.105,.018,'Basal B: per doubling. Reactivity E: per 1-unit increase (doubling of A).  Minimal: age + SOFA; extended: + MAP, lactate, vasopressors.',fontsize=9,color=MUTED)
 save(fig,3)
def tag(ax,letter,title,x=-.10,y=1.10):
    ax.text(x,y,letter,transform=ax.transAxes,fontsize=19,fontweight='bold',va='bottom',clip_on=False)
    ax.set_title(title,loc='left',fontweight='bold',pad=13)
def simple(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=3,width=.7)
def kde_line(ax,values,color,vertical=False):
    v=np.asarray(values);v=v[np.isfinite(v)]
    if len(v)>3 and np.std(v)>0:
        x=np.linspace(v.min()-.15*v.std(),v.max()+.15*v.std(),200);y=gaussian_kde(v)(x)
        if vertical:ax.fill_betweenx(x,0,y,color=color,alpha=.22,lw=0);ax.plot(y,x,color=color,lw=1)
        else:ax.fill_between(x,0,y,color=color,alpha=.22,lw=0);ax.plot(x,y,color=color,lw=1)
def ribbon(ax,x0,x1,y0,y1,h,color,alpha=.5):
    verts=[(x0,y0),(x0+.45*(x1-x0),y0),(x1-.45*(x1-x0),y1),(x1,y1), (x1,y1+h),(x1-.45*(x1-x0),y1+h),(x0+.45*(x1-x0),y0+h),(x0,y0+h),(x0,y0)]
    codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO, MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts,codes),facecolor=color,edgecolor='none',alpha=alpha))
def alluvial(ax,q):
    n=len(q);gap=8
    n0=[int(q.state1.eq(s).sum()) for s in STATES];n1=[int(q.state2.eq(s).sum()) for s in STATES]
    n2=[int(q.death.eq(x).sum()) for x in [0,1]]
    total=n+gap*3
    def bottoms(counts):
        cur=total;res=[]
        for count in counts:cur-=count;res.append(cur);cur-=gap
        return np.array(res,dtype=float)
    b0,b1,b2=bottoms(n0),bottoms(n1),bottoms(n2)
    b2-=gap
    u0=b0.copy();u1=b1.copy()
    for a,s in enumerate(STATES):
        for b,t in enumerate(STATES):
            nn=int((q.state1.eq(s)&q.state2.eq(t)).sum())
            if nn:ribbon(ax,.13,.48,u0[a],u1[b],nn,SC[a],.43);u0[a]+=nn;u1[b]+=nn
    u1=b1.copy();u2=b2.copy()
    for a,s in enumerate(STATES):
        for b,death in enumerate([0,1]):
            nn=int((q.state2.eq(s)&q.death.eq(death)).sum())
            if nn:ribbon(ax,.52,.87,u1[a],u2[b],nn,[SURV,DEAD][b],.40);u1[a]+=nn;u2[b]+=nn
    for x,base,counts,cols in [(.10,b0,n0,SC),(.48,b1,n1,SC),(.87,b2,n2,[SURV,DEAD])]:
        for j,(b,nj,c) in enumerate(zip(base,counts,cols)):
            ax.add_patch(Rectangle((x,b),.04,nj,facecolor=c,edgecolor='white',lw=.6))
            ax.text(x+.02,b+nj/2,str(nj),ha='center',va='center',fontsize=7.1 if nj>99 else 8.6,color='white',fontweight='bold')
    for j,label in enumerate(SHORT):ax.text(.07,b0[j]+n0[j]/2,label,ha='right',va='center',fontsize=8)
    for j,label in enumerate(['Survived','Died']):ax.text(.94,b2[j]+n2[j]/2,label,ha='left',va='center',fontsize=9,color=[SURV,DEAD][j])
    for x,label in [(.12,'Day 1'),(.50,'Day 2'),(.89,'Outcome')]:ax.text(x,total+11,label,ha='center',fontweight='bold',fontsize=10)
    ax.set_xlim(-.27,1.12);ax.set_ylim(-8,total+24);ax.axis('off')
def make4():
    q=D.dropna(subset=['x1','E1','x2','E2']).copy()
    fig=plt.figure(figsize=(13.5,12.2));gs=fig.add_gridspec(3,2,height_ratios=[1.16,1,1.02],wspace=.40,hspace=.57, left=.075,right=.965,top=.945,bottom=.095)
    ax=fig.add_subplot(gs[0,0]);tag(ax,'A','Observed movement through a fixed state space')
    gx=np.linspace(-7.0,4.3,48);gy=np.linspace(-3.3,5.9,40);xx,yy=np.meshgrid(gx,gy)
    X=q.x1.to_numpy();Y=q.E1.to_numpy();dx=q.dx.to_numpy();de=q.dE.to_numpy()
    w=np.exp(-.5*((xx[...,None]-X)/1.05)**2-.5*((yy[...,None]-Y)/.80)**2)
    den=w.sum(2);u=(w*dx).sum(2)/den;v=(w*de).sum(2)/den
    mask=den<4;u=np.ma.masked_where(mask,u);v=np.ma.masked_where(mask,v)
    ax.streamplot(gx,gy,u,v,color='#B5C3CD',density=.85,linewidth=.75,arrowsize=.7,zorder=0)
    for outcome,c in [(0,SURV),(1,DEAD)]:
        z=q[q.death.eq(outcome)]
        ax.quiver(z.x1,z.E1,z.dx,z.dE,angles='xy',scale_units='xy',scale=1,width=.0022, headwidth=3.3,headlength=4.3,color=c,alpha=.27,zorder=1)
        ax.scatter(z.x2,z.E2,s=13,color=c,alpha=.72,edgecolors='white',linewidths=.2,label=f'{"Survived" if outcome==0 else "Died"} (n={len(z)})',zorder=3)
    ax.axvline(np.log2(CUT),color=MUTED,ls='--',lw=.85);ax.axhline(0,color=MUTED,ls='--',lw=.85)
    ax.set(xlim=(-7.0,4.6),ylim=(-3.4,6.3),xlabel=r'Basal perfusion, $\log_2 B$',ylabel=r'Baseline-adjusted reactivity, $E$')
    ax.legend(loc='upper right',fontsize=8.2,handletextpad=.4,labelspacing=.35)
    ax.text(.03,.94,'218 paired observations',transform=ax.transAxes,fontsize=8.5,color=MUTED)
    simple(ax)
    ax=fig.add_subplot(gs[0,1]);tag(ax,'B','Where patients moved, and their recorded outcome',x=-.12)
    alluvial(ax,q)
    sub=GridSpecFromSubplotSpec(2,2,subplot_spec=gs[1,0],height_ratios=[.20,1],width_ratios=[1,.18],hspace=.03,wspace=.03)
    ax=fig.add_subplot(sub[1,0]);top=fig.add_subplot(sub[0,0],sharex=ax);right=fig.add_subplot(sub[1,1],sharey=ax)
    tag(top,'C','Basal change tracks peripheral refill recovery',y=1.5)
    pp=T['partial_crt_points'];curve=T['partial_crt_curve']
    for o,c in [(0,SURV),(1,DEAD)]:
        p=pp[pp.death.eq(o)];ax.scatter(p.x_residual,p.y_residual,s=15,color=c,alpha=.55,edgecolor='white',lw=.25)
        kde_line(top,p.x_residual,c);kde_line(right,p.y_residual,c,True)
    ax.fill_between(curve.x,curve.lower95,curve.upper95,color=BLUE,alpha=.16,lw=0)
    ax.plot(curve.x,curve.effect,color=BLUE,lw=1.8)
    ax.axhline(0,color=MUTED,lw=.6,ls=':');ax.axvline(0,color=MUTED,lw=.6,ls=':')
    ax.set(xlabel=r'Adjusted $\Delta\log_2 B$',ylabel=r'Adjusted $\Delta$CRT (s)')
    r=ASSOC.query('model == "day2_CRT_trajectory" and term == "dx"').iloc[0]
    ax.text(.04,.96,f'{r.effect:.3f} s per doubling\n95% CI {r.lower95:.3f} to {r.upper95:.3f}',transform=ax.transAxes,va='top',fontsize=9,
      bbox=dict(boxstyle='round,pad=.3',facecolor='white',edgecolor='none',alpha=.84))
    top.axis('off');right.axis('off');simple(ax)
    ax=fig.add_subplot(gs[1,1]);tag(ax,'D','The declining-ratio group contains divergent changes',x=-.12)
    paradox=q[(q.dx>0)&(q.dy>0)&(q.dr<0)]
    features=[('Basal perfusion B','dx'),('Sampled peak P','dy'),('Rebound ratio R','dr'),('CRT','dCRT'),('Lactate','dloglac'),('Norepinephrine','dNE_observed')]
    counts=[]
    for i,(label,col) in enumerate(features):
        vals=paradox[col].dropna().to_numpy();n=len(vals);less=int((vals < -1e-10).sum());same=int((abs(vals)<=1e-10).sum());more=n-less-same
        counts.append({'measure':label,'n':n,'decreased':less,'unchanged':same,'increased':more})
        left=-100*less/n
        ax.barh(i,100*less/n,left=left,color=BLUE,height=.58,alpha=.86)
        ax.barh(i,100*same/n,left=0,color='#C8D1D7',height=.58)
        ax.barh(i,100*more/n,left=100*same/n,color=DEAD,height=.58,alpha=.86)
        if less:ax.text(left/2,i,str(less),va='center',ha='center',color='white',fontsize=9,fontweight='bold')
        if same:ax.text(50*same/n,i,str(same),va='center',ha='center',fontsize=9,fontweight='bold')
        if more:ax.text(100*same/n+50*more/n,i,str(more),va='center',ha='center',color='white',fontsize=9,fontweight='bold')
        ax.text(108,i,f'n={n}',va='center',fontsize=8.5,color=MUTED)
    ax.set_yticks(range(6),[x[0] for x in features]);ax.set_ylim(5.7,-.8);ax.set_xlim(-104,124)
    ax.set_xticks([-100,-50,0,50,100],['100','50','0','50','100']);ax.axvline(0,color=INK,lw=.65)
    ax.set_xlabel('Patients with measurements (%)')
    ax.legend(handles=[Rectangle((0,0),1,1,color=c,label=l) for c,l in [(BLUE,'Decreased'),('#C8D1D7','Unchanged'),(DEAD,'Increased')]],
        ncol=3,loc='upper center',bbox_to_anchor=(.5,1.04),fontsize=8,handlelength=1,columnspacing=1)
    simple(ax)
    sub=GridSpecFromSubplotSpec(1,2,subplot_spec=gs[2,:],width_ratios=[1.55,1],wspace=.46)
    ax=fig.add_subplot(sub[0]);tag(ax,'E','Reactivity change and subsequent mortality',x=-.075)
    model=ASSOC[ASSOC.model.eq('mortality_trajectory_landmark_day2')].set_index('term')
    boot=T['mortality_trajectory_refit_bootstrap'].set_index('term')
    rows=['dx','dE'];labels=['Basal change\n'+r'$\Delta\log_2 B$','Reactivity\n'+r'$\Delta E$']
    for i,(term,c) in enumerate(zip(rows,[BLUE,TEAL])):
        r=model.loc[term];b=boot.loc[term]
        ax.plot([b.OR_lower95,b.OR_upper95],[i+.06]*2,color=c,alpha=.26,lw=7,solid_capstyle='butt')
        ax.errorbar(r.effect,i,xerr=[[r.effect-r.lower95],[r.upper95-r.effect]],fmt='o',color=c,lw=1.7,capsize=4,ms=7)
        ax.text(1.52,i,f'{r.effect:.2f} ({r.lower95:.2f}–{r.upper95:.2f})',va='center',fontsize=10,color=c)
    ax.set_xscale('log');ax.set_xlim(.15,3.9);ax.set_ylim(1.75,-.75)
    ax.set_yticks([0,1],labels);ax.axvline(1,color=MUTED,lw=.9,ls='--')
    ax.set_xticks([.2,.4,.6,1,2],['0.2','0.4','0.6','1','2']);ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel('Odds ratio per 1-unit coordinate change')
    ax.text(1.52,-.52,'OR (Wald 95% CI)',fontsize=9,fontweight='bold')
    ax.legend(handles=[Line2D([0],[0],color=MUTED,lw=1.8,label='Wald 95% CI'),Line2D([0],[0],color=MUTED,lw=7,alpha=.3,label='Refitted bootstrap 95% CI')], loc='lower left',fontsize=8.5)
    simple(ax)
    ax=fig.add_subplot(sub[1]);loo=T['trajectory_leave_one_out'].OR_dE
    ax.hist(loo,bins=22,color=TEAL,alpha=.28,density=True,edgecolor='white',lw=.5)
    xx=np.linspace(loo.min()-.02,loo.max()+.02,300);ax.plot(xx,gaussian_kde(loo)(xx),color=TEAL,lw=2)
    ax.axvline(model.loc['dE','effect'],color=INK,ls='--',lw=1.2)
    ax.axvspan(loo.min(),loo.max(),alpha=.035,color=TEAL)
    ax.set(xlabel=r'Leave-one-out OR for $\Delta E$',ylabel='Density',xlim=(.34,.53))
    ax.set_title('All 208 leave-one-patient-out fits',loc='left',fontweight='bold',pad=13)
    ax.text(.04,.93,f'Observed range\n{loo.min():.3f}–{loo.max():.3f}',transform=ax.transAxes,va='top',fontsize=9)
    simple(ax)
    fig.text(.075,.022,'Outcome: recorded in-hospital death through day 28. Mortality trajectory set: 208 patients / 73 deaths; paired CRT: 217.\nE is on the log₂ scale: a 1-unit increase corresponds to a doubling of the fold-reference factor A.',fontsize=8.5,color=MUTED)
    save(fig,4)
def bias(ax,which):
    g=np.linspace(-2,2,151);x,y=np.meshgrid(g,g);z=coupling(x,y,BETA)[0 if which=='R' else 1]
    lev=np.linspace(-4,4,25);im=ax.contourf(x,y,z,levels=lev,cmap='RdBu_r',extend='both')
    cc=ax.contour(x,y,z,levels=[-3,-2,-1,0,1,2,3],colors='white',linewidths=.55,alpha=.8)
    ax.clabel(cc,inline=True,fontsize=8,fmt='%g')
    ax.plot(g,g,color=INK,ls='--',lw=1.3,label='Common coupling')
    ax.scatter([0],[0],s=28,facecolors='white',edgecolors=INK,lw=1,zorder=5)
    ax.set_xticks([-2,-1,0,1,2],['0.25','0.5','1','2','4']);ax.set_yticks([-2,-1,0,1,2],['0.25','0.5','1','2','4'])
    ax.set_xlabel(r'Basal effective coupling, $\kappa_B$');ax.set_ylabel(r'Peak effective coupling, $\kappa_P$')
    ax.text(.04,.94,r'$\Delta\log_2R=\log_2\kappa_P-\log_2\kappa_B$' if which=='R' else r'$\Delta E=\log_2\kappa_P-\beta\log_2\kappa_B$',
       transform=ax.transAxes,va='top',fontsize=10,bbox=dict(boxstyle='round,pad=.25',facecolor='white',edgecolor='none',alpha=.92))
    return im
def rain(ax,vals,positions,colorset):
    rng=np.random.default_rng(1301)
    for v,x,c in zip(vals,positions,colorset):
        if np.std(v)>1e-12:
            kde=gaussian_kde(v);y=np.linspace(min(v)-1,max(v)+1,220);density=kde(y);density=.34*density/density.max()
            ax.fill_betweenx(y,x,x-density,color=c,alpha=.28,lw=0);ax.plot(x-density,y,color=c,lw=.8)
        ax.scatter(x+.07+rng.uniform(0,.20,len(v)),v,s=6,color=c,alpha=.32,edgecolor='none')
        bp=ax.boxplot([v],positions=[x],widths=.1,patch_artist=True,showfliers=False,manage_ticks=False,
           boxprops=dict(facecolor='white',edgecolor=c,lw=1),medianprops=dict(color=c,lw=1.5),
           whiskerprops=dict(color=c,lw=.8),capprops=dict(color=c,lw=.8))
def monitor(ax):
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    bg='#132939';card='#20394B';edge='#416070';white='#E9F2F6';muted='#AEC3CF';accent='#68C8CA'
    ax.add_patch(FancyBboxPatch((.01,.015),.98,.97,boxstyle='round,pad=.008,rounding_size=.025',facecolor=bg,edgecolor='#102331',lw=1))
    ax.text(.04,.91,'BASELINE-AWARE OPTICAL PERFUSION',color=white,weight='bold',fontsize=14)
    ax.text(.96,.92,'ILLUSTRATIVE UI  |  Patient D152',color=muted,ha='right',fontsize=9)
    ax.text(.04,.845,'Basal state and reactivity stay visible together',color=muted,fontsize=10)
    inner=ax.inset_axes([.055,.31,.35,.45]);inner.set_facecolor(card)
    inner.axvspan(-4,np.log2(CUT),color='#406D83',alpha=.3);inner.axhspan(0,2,color='#297B79',alpha=.15)
    inner.axvline(np.log2(CUT),color=muted,lw=.7,ls='--');inner.axhline(0,color=muted,lw=.7,ls='--')
    patient=D[D.patient_key.eq('D152')].iloc[0]
    inner.annotate('',xy=(patient.x2,patient.E2),xytext=(patient.x1,patient.E1),arrowprops=dict(arrowstyle='-|>',color=accent,lw=2,mutation_scale=14))
    inner.scatter([patient.x1],[patient.E1],s=60,facecolors=card,edgecolors=accent,lw=1.4,zorder=5)
    inner.scatter([patient.x2],[patient.E2],s=70,color=accent,edgecolors=white,lw=.7,zorder=5)
    inner.text(patient.x1-.15,patient.E1+.32,'Day 1',color=white,fontsize=8,ha='center')
    inner.text(patient.x2,patient.E2-.42,'Day 2',color=white,fontsize=8,ha='center')
    inner.set(xlim=(-4,4.6),ylim=(-1.7,1.8),xlabel='Basal perfusion B',ylabel='Adjusted reactivity E')
    inner.set_xticks([-3,-1,1,3],['0.125','0.5','2','8']);inner.set_yticks([-1,0,1]);inner.tick_params(colors=muted,labelsize=8)
    inner.xaxis.label.set_color(muted);inner.yaxis.label.set_color(muted)
    for sp in inner.spines.values():sp.set_color(edge)
    fields=[('BASAL PERFUSION  B',f'{patient.B2:.2f}',f'Day 1: {patient.B1:.2f}'),
            ('SAMPLED PEAK  P',f'{patient.P2:.2f}',f'Day 1: {patient.P1:.2f}'),
            ('ADJUSTED FACTOR  A',f'{patient.A2:.2f}×',f'E = {patient.E2:+.3f} log₂'),
            ('CONVENTIONAL RATIO  R',f'{patient.R2:.2f}×',f'Relative rebound: {100*(patient.R2-1):.1f}%')]
    for i,(name,val,note) in enumerate(fields):
        x=.455+(i%2)*.25;y=.50-(i//2)*.28
        ax.add_patch(FancyBboxPatch((x,y),.22,.24,boxstyle='round,pad=.01,rounding_size=.015',facecolor=card,edgecolor=edge,lw=.6))
        ax.text(x+.015,y+.195,name,fontsize=8,color=muted,weight='bold')
        ax.text(x+.015,y+.092,val,fontsize=25,color=white,weight='bold')
        ax.text(x+.015,y+.025,note,fontsize=8.5,color=muted)
    pills=[(.04,.32,'30 s snapshots',accent),(.38,.26,'Contact QC: unassessed','#A8B6C0'),(.66,.30,'Reference: Brazil23 · Day 1',muted)]
    for x,w,label,c in pills:
        ax.add_patch(FancyBboxPatch((x,.07),w,.075,boxstyle='round,pad=.01,rounding_size=.02',facecolor=card,edgecolor=edge,lw=.6))
        ax.add_patch(Circle((x+.018,.108),.005,color=c));ax.text(x+.032,.108,label,fontsize=9,color=white,va='center')
def make5():
    fig=plt.figure(figsize=(13.5,12.6));gs=fig.add_gridspec(3,2,height_ratios=[.95,1.12,1.10],wspace=.36,hspace=.57, left=.08,right=.965,top=.945,bottom=.055)
    ax=fig.add_subplot(gs[0,0]);tag(ax,'A','Ratio bias across the effective coupling plane');im=bias(ax,'R')
    ax=fig.add_subplot(gs[0,1]);tag(ax,'B','Adjusted reactivity has a different bias geometry');bias(ax,'E')
    cb=fig.colorbar(im,ax=fig.axes[:2],location='right',fraction=.023,pad=.025,shrink=.95);cb.set_label('Bias (log₂ units)',fontsize=9)
    ax.legend(loc='lower right',fontsize=8)
    sub=GridSpecFromSubplotSpec(1,2,subplot_spec=gs[1,0],wspace=.20)
    axs=[fig.add_subplot(sub[j]) for j in range(2)]
    tag(axs[0],'C','Prediction depends on the available clinical context',x=-.31)
    met=T['prediction_metrics']
    names=['optical_R','optical_BE','macro_R','macro_BE','clinical','clinical_BP','clinical_BP_time']
    labels=['Optical: R','Optical: B + E','Age/MAP + R','Age/MAP + B + E','Full clinical','Full + B + P','Full + BP + time']
    tasks=[('mortality',BLUE,'o',-.12),('nextday_CRT_MAP65',TEAL,'s',.12)]
    for j,metric in enumerate(['brier','auc']):
        ax=axs[j]
        for task,c,marker,offset in tasks:
            table=met[met.task.eq(task)].set_index('model')
            for i,name in enumerate(names):
                r=table.loc[name];lo=r[f'{metric}_lower95'];hi=r[f'{metric}_upper95']
                ax.errorbar(r[metric],i+offset,xerr=[[r[metric]-lo],[hi-r[metric]]],fmt=marker,color=c,ms=3.5,lw=.85,capsize=1.6)
            ax.axvline(table.loc['clinical',metric],color=c,ls='--',lw=.7,alpha=.6)
        for yy in [1.5,3.5]:ax.axhline(yy,color='#D3DCE2',lw=.7)
        ax.set_ylim(6.7,-.7);ax.set_yticks(range(len(names)),labels if j==0 else [])
        ax.set_xlabel('Brier score ↓' if j==0 else 'AUROC ↑')
        ax.tick_params(axis='y',length=0,labelsize=8.5)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(3));simple(ax)
    axs[0].legend(handles=[Line2D([0],[0],color=BLUE,marker='o',lw=.8,label='Mortality'),Line2D([0],[0],color=TEAL,marker='s',lw=.8,label='Next-day CRT >3 s')],loc='upper left',bbox_to_anchor=(-.42,-.17),ncol=2,fontsize=7.8,columnspacing=1.5,handlelength=1.5)
    sub=GridSpecFromSubplotSpec(1,2,subplot_spec=gs[1,1],width_ratios=[.9,1.1],wspace=.40)
    ax=fig.add_subplot(sub[0]);tag(ax,'D','Sparse grids miss observed peaks',x=-.28)
    seq=D[[f'pi_{t}s_d1' for t in range(30,301,30)]].to_numpy();peak=seq.max(1)
    loss={};phases={}
    for step in [2,3]:
        peaks=np.stack([seq[:,p::step].max(1) for p in range(step)],1)
        phases[step]=100*(1-peaks/peak[:,None]);loss[step]=phases[step].mean(1)
    rain(ax,[np.zeros(len(D)),loss[2],loss[3]],[0,1,2],['#AAB7C1',BLUE,TEAL])
    for x,step in [(1,2),(2,3)]:
        pct=np.percentile(loss[step],95);ax.plot([x-.3,x+.3],[pct]*2,color=INK,ls=':',lw=1)
        ax.text(x-.14,pct+2,f'{pct:.1f}%',ha='center',fontsize=8.7,color=[BLUE,TEAL][step-2],fontweight='bold')
    ax.text(.04,.97,'95th percentile',transform=ax.transAxes,va='top',fontsize=8,color=MUTED)
    ax.set(xlim=(-.5,2.5),ylim=(-2,68),ylabel='Peak loss (%)');ax.set_xticks([0,1,2],['30 s','60 s','90 s']);simple(ax)
    target=np.percentile(loss[3],75);idx=int(np.argmin(abs(loss[3]-target)));r=D.iloc[idx]
    ax=fig.add_subplot(sub[1]);t=np.arange(30,301,30)
    ax.plot(t,seq[idx],'-o',color=INK,ms=3.4,lw=1.5,label='30 s observed')
    for step,c,marker in [(2,BLUE,'s'),(3,TEAL,'D')]:
        ax.plot(t[::step],seq[idx,::step],ls='--',marker=marker,ms=4,lw=1,color=c,alpha=.85,label=f'{30*step} s grid')
    ax.axhline(peak[idx],color=INK,ls=':',lw=.9)
    ax.set(xlabel='After release (s)',ylabel='Measured PI',xlim=(20,310));ax.set_xticks([30,120,210,300])
    ax.set_title(f'Observed patient {r.patient_key}',fontweight='bold',fontsize=9.5,pad=9)
    ax.legend(loc='best',fontsize=7.8,handlelength=1.8);simple(ax)
    OUT.mkdir(exist_ok=True)
    ax=fig.add_subplot(gs[2,:]);tag(ax,'E','A monitor that preserves basal state, reactivity and acquisition context',x=-.04,y=1.04);monitor(ax)
    fig.text(.08,.016,'Coupling panels are analytical sensitivity surfaces. UI values are observed cohort measurements; continuous acquisition and contact quality were not measured.',fontsize=8.3,color=MUTED)
    save(fig,5)
def flow(d):
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.set(xlim=(0, 10), ylim=(0, 8)); ax.axis('off')
    boxes = [(3.2, 6.8, 3.6, .8, 'Full cohort: 226 patients\n90 deaths; 136 survivors'),
             (7.3, 5.3, 2.6, 1, '8 excluded; all died\n7 missing day-2 records\n1 day-2 basal PI = 0'),
             (3.2, 4.9, 3.6, .9, 'Paired optical records: 218\n82 deaths; 136 survivors'),
             (.1, 2.6, 2.8, 1.1, 'Paired CRT: 217\n1 incomplete CRT pair'),
             (3.6, 2.6, 2.8, 1.1, 'Paired lactate: 175\n43 incomplete lactate pairs'),
             (7.1, 2.6, 2.8, 1.1, 'Outcome time >2 days: 208\n73 deaths; 135 survivors')]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.1', facecolor='#EDF3F7', edgecolor='#537389', lw=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    ax.text(8.5,1.9,'Excluded from the >2-day set:\n9 deaths and 1 survivor with\nrecorded outcome time ≤2 days',ha='center',va='top',fontsize=9)
    for start, stop in [((5,6.7),(5,5.9)),((5,6.3),(7.2,5.8)),((5,4.8),(1.5,3.85)),((5,4.8),(5,3.85)),((5,4.8),(8.5,3.85))]:
        ax.annotate('',xy=stop,xytext=start,arrowprops=dict(arrowstyle='->',color='#537389',lw=1.2))
    ax.set_title('Figure S1. Patient flow and analysis-specific denominators',loc='left',weight='bold',pad=15)
    save(fig,'S1')
def render_all(data, tables, directory):
    global D,T,A,ASSOC,AL,DM,ST,OUT,alpha,beta,BETA,cut,CUT,xc,ORDER,COLS,CM,STATES,SC,SURV,DEAD,INK,MUTED,BLUE,TEAL,SHORT,LABELS,RNG
    D,T,OUT=data,tables,Path(directory); OUT.mkdir(parents=True,exist_ok=True)
    A=ASSOC=T['associations']; AL=T['allometry']; DM=T['deming_sensitivity']; ST=T['day1_states'].set_index('state')
    alpha,beta=AL.iloc[0][['alpha','beta']]; BETA=beta; cut=CUT=D.B1.median(); xc=np.log2(cut)
    ORDER=STATES; COLS=SC=['#C15D67','#E3A33F','#527C9E','#3B9B8E']; CM=dict(zip(ORDER,COLS))
    SURV,DEAD,INK,MUTED,BLUE,TEAL='#0072B2','#D55E00','#23323F','#647584','#3C729E','#208C87'
    LABELS=['Low B\nBelow expected','Low B\nAbove expected','High B\nBelow expected','High B\nAbove expected']
    SHORT=['Low B / E < 0','Low B / E ≥ 0','High B / E < 0','High B / E ≥ 0']; RNG=np.random.default_rng(712041)
    controls=['age','sofa_d1','crt_sec_d1','x1','E1','dE']; q=D.dropna(subset=controls+['dx','dCRT'])
    Z=np.c_[np.ones(len(q)),q[controls]]; x=q.dx.to_numpy(); y=q.dCRT.to_numpy()
    xr=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]; yr=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
    T['partial_crt_points']=pd.DataFrame(dict(x_residual=xr,y_residual=yr,death=q.death))
    coefficient=A.query('model=="day2_CRT_trajectory" and term=="dx"').iloc[0]; xx=np.linspace(np.quantile(xr,.01),np.quantile(xr,.99),150)
    lo,hi=coefficient.lower95*xx,coefficient.upper95*xx
    T['partial_crt_curve']=pd.DataFrame(dict(x=xx,effect=coefficient.effect*xx,lower95=np.minimum(lo,hi),upper95=np.maximum(lo,hi)))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white','legend.frameon':False})
    for function in [fig1,fig2,fig3,make4,make5]:
        function()
    flow(D)
