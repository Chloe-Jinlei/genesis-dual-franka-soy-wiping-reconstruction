"""Make review artifacts exclusively from the saved simulated states and frames."""
from pathlib import Path
import json, zipfile, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/final'

def annotation_font(size):
    """Keep the original appearance when DejaVu is installed."""
    for font in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(font, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def caution_description(validation):
    """Describe the saved measurements instead of assuming a particular run."""
    cautions=validation.get('cautions', {})
    if not cautions:
        return '本次采样未记录数值警示；这不等于无误差或真实材料验证。'
    details=[]
    extrema=validation.get('extrema', {})
    thresholds=validation.get('thresholds', {})
    for name, event in cautions.items():
        detail=(f"`{name}`：{event['count']} 个采样点，"
                f"首次 {event['first_time_s']:.2f} s，最后 {event['last_time_s']:.2f} s")
        if name=='cloth_stretch_caution' and 'cloth_edge_ratio_max' in extrema:
            peak=(extrema['cloth_edge_ratio_max']['value']-1)*100
            threshold=(thresholds['cloth_max_edge_ratio_caution']-1)*100
            detail+=f'；最大局部布边伸长 {peak:.3f}%，自设警戒值 {threshold:g}%'
        elif name=='acceleration_caution':
            accelerations=np.asarray(validation.get('actual_max_abs_joint_acceleration_after_startup', []))
            if accelerations.size:
                peak=float(np.max(accelerations[..., :7]))
                detail+=(f"；臂关节加速度峰值 {peak:.3f} rad/s²，"
                         f"自设警戒值 {thresholds['acceleration_caution_rad_s2']:g} rad/s²")
        details.append('- '+detail+'。')
    text=f'本次保留 {len(cautions)} 类数值警示：\n\n'+'\n'.join(details)
    text+='\n\n警示与失败项分开记录；应结合 `validation.json` 的失败项和阈值解释，不能把“通过”解释为无误差或真实材料验证。'
    if 'cloth_stretch_caution' in cautions:
        text+='图中的红叉标出采样记录的首次布料警示点。'
    return text

def bundle_files():
    """Include runnable sources, setup/docs, inputs and final evidence only."""
    root_suffixes={'.py','.sh','.md','.json','.toml','.txt','.yaml','.yml','.cfg','.ini'}
    root_names={'.gitignore','.gitattributes','Dockerfile','Makefile','LICENSE','NOTICE'}
    files=[p for p in ROOT.iterdir() if p.is_file() and
           (p.suffix.lower() in root_suffixes or p.name in root_names)]
    for directory in (ROOT/'reference_images', ROOT/'assets', OUT):
        if directory.exists():
            files.extend(p for p in directory.rglob('*') if p.is_file())
    return sorted({p for p in files
                   if p.name!='file_manifest.json'
                   and p.suffix.lower() not in {'.zip','.pyc','.pyo'}
                   and not any(part in {'__pycache__','.git','.venv'} for part in p.relative_to(ROOT).parts)
                   and p.resolve().is_relative_to(ROOT)},
                  key=lambda p:p.relative_to(ROOT).as_posix())

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    validation=json.loads((OUT/'validation.json').read_text())
    configuration=json.loads((OUT/'configuration.json').read_text())
    records=json.loads((OUT/'metrics.json').read_text())
    t=np.array([r['time_s'] for r in records])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    axes[0].plot(t,[r['free_liquid_mass_kg']*1000 for r in records],color='#8b4526',label='Soy sauce on table')
    axes[0].plot(t,[r['absorbed_liquid_mass_kg']*1000 for r in records],color='#087f8c',label='Soy sauce in cloth')
    axes[0].plot(t,[r['total_liquid_mass_kg']*1000 for r in records],color='#343c49',ls='--',label='Total liquid')
    axes[0].set_ylabel('Liquid mass (g)');axes[0].legend(ncol=3,loc='center left')
    axes[1].plot(t,[(r['cloth_edge_stretch_max']-1)*100 for r in records],color='#bc6138',label='Maximum edge extension')
    axes[1].plot(t,[(r['cloth_edge_stretch_p95']-1)*100 for r in records],color='#688b72',label='95th percentile')
    stretch_threshold=(validation['thresholds']['cloth_max_edge_ratio_caution']-1)*100
    axes[1].axhline(stretch_threshold,ls=':',color='#999',label='Declared caution threshold')
    caution=validation['cautions'].get('cloth_stretch_caution')
    if caution:
        axes[1].scatter([caution['first_time_s']],[(caution['first_value']-1)*100],
                        marker='x',color='#b52f34',zorder=5)
    axes[1].set_ylabel('Cloth extension (%)');axes[1].legend(ncol=3,loc='upper left')
    axes[2].plot(t,[r['joint_speed_max_rad_s'] for r in records],color='#345e8d')
    axes[2].set_ylabel('Max arm joint speed (rad/s)');axes[2].set_xlabel('Simulated time (s)')
    lift=summary['clean_trigger_time_s']
    for ax in axes:
        ax.grid(alpha=.15)
        for x in (8.5,14,17,lift):ax.axvline(x,color='#aaa',lw=.6,alpha=.5)
    backend=configuration.get('backend','unknown backend')
    fig.suptitle(f'Measured simulation checks | Genesis + conservative film / wicking | {backend}',fontsize=13)
    fig.savefig(OUT/'physics_checks.png',dpi=180);plt.close(fig)
    paths=[OUT/'closeup_000.0_initial.png',OUT/'closeup_011.0_lift_cloth.png',
           OUT/'closeup_020.0_wipe.png',OUT/f"closeup_{summary['completed_simulation_time_s']:05.1f}_hold_wet_cloth.png"]
    labels=['1. Initial cloth and soy sauce','2. Two arms lift the cloth',
            '3. Contact, wiping and absorption','4. Wet cloth lifted; measured residue below threshold']
    canvas=Image.new('RGB',(1600,1280),(244,245,247));d=ImageDraw.Draw(canvas)
    font=annotation_font(20)
    for k,(p,label) in enumerate(zip(paths,labels)):
        im=Image.open(p).convert('RGB').resize((780,585),Image.Resampling.LANCZOS)
        x=10+(k%2)*800;y=10+(k//2)*640
        canvas.paste(im,(x,y));d.text((x+4,y+595),label,font=font,fill=(30,46,57))
    canvas.save(OUT/'process_preview.png')
    final=summary['final'];cautions=list(validation['cautions'])
    caution_text=caution_description(validation)
    clean_threshold=validation['thresholds']['residual_fraction_success']
    clean_status='低于清洁阈值' if final['residual_fraction']<=clean_threshold else '高于清洁阈值'
    text=f'''# 本次 GPU 仿真结果

已完成 {summary['completed_simulation_time_s']:.2f} 秒的连续仿真，往复擦拭 {summary['wipe_cycles']} 个完整周期；在 {summary['clean_trigger_time_s']:.2f} 秒测得剩余量达到阈值后才启动最终提布。

| 核验项 | 实际记录 |
| --- | --- |
| 设备 | {configuration.get('device','未记录')} / {backend} |
| 初始酱油质量 | {final['initial_liquid_mass_kg']*1000:.6f} g |
| 最终桌面剩余（模型数值） | {final['free_liquid_mass_kg']*1000:.3e} g，{clean_status} |
| 最终布内液体 | {final['absorbed_liquid_mass_kg']*1000:.6f} g |
| 液体相对质量误差最大值 | {summary['max_liquid_relative_mass_error']:.3e} |
| 最终湿布最低处离桌 | {summary['final_lift_clearance_m']*100:.3f} cm |
| 100 Hz 数值检查 | {'通过' if validation['sampled_numerical_checks_passed'] else '未通过'} |
| 所列任务判据 | {'通过' if validation['task_success_under_stated_criteria'] else '未通过'} |
| 警示项 | {', '.join(cautions) if cautions else '无'} |

清洁阈值是桌面剩余量不超过初始质量的 {clean_threshold*100:g}%，并且至少 {(1-clean_threshold)*100:g}% 进入布中。以上数字不是数学上零残留的声明。完整的 100 Hz 极值、警示次数和工程阈值请查看 `validation.json`；绘图与 `metrics.csv` 每 0.1 秒采样。

{caution_text}

## 查看

- `simulation_overview.mp4`：完整动作全景。
- `simulation_top.mp4`：俯视核对连续路径。
- `simulation_closeup.mp4`：布料变形、接触、染色和提布近景。
- `process_preview.png`：四个过程画面。
- `physics_checks.png`：质量守恒、布料变形及实际关节速度。
- `state_samples.npz`：实际布粒子、液膜和关节的状态采样。

## 解释范围

这是一套独立、可复现的物理近似重建，未访问原仿真过程或历史记忆。初始布局来自本次图片与 assets；未观测参数是显式假设。机械臂及布料由 Genesis 动力学求解，液体采用守恒薄膜、接触吸收和织物内扩散模型。抓取采用理想无滑移约束，尚未验证真实夹爪摩擦和完整流固反作用；材料参数未经实验标定。数值判据通过不等于真实物理实验验证或精确恢复原仿真。
'''
    (OUT/'RESULTS.md').write_text(text)
    files=bundle_files()
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT/'file_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    with zipfile.ZipFile(ROOT/'genesis_reconstruction_bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[OUT/'file_manifest.json']:z.write(p,Path(ROOT.name)/p.relative_to(ROOT))
    print(json.dumps({'clean':summary['clean'],'validated':validation['task_success_under_stated_criteria'],
        'duration':summary['completed_simulation_time_s'],'bundle_bytes':(ROOT/'genesis_reconstruction_bundle.zip').stat().st_size},indent=2))

if __name__=='__main__':main()
