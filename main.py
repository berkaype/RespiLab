import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. VERİLER
# ==========================================
our_data_real = np.array([
    54.0, 55.2, 56.0, 55.6, 44.6, 32.0, 29.0, 26.0, 25.0, 25.5, 25.0, 25.2,
    25.1, 24.2, 24.7, 24.0, 23.7, 23.8, 23.0, 22.5, 22.7, 21.5, 20.6, 19.6,
    19.6, 19.5, 18.1, 17.5, 15.6, 15.2, 13.2, 12.0, 10.0, 9.1, 8.3, 6.6,
    6.1, 5.5, 5.5, 5.2, 5.1, 5.0, 4.6, 4.9, 5.1, 4.4, 4.5, 4.4,
    4.3, 4.3, 4.3, 4.2, 4.1, 4.0, 3.9, 3.9, 3.8, 3.8, 3.7, 3.6
])
t_meas = np.linspace(0, 1.0, len(our_data_real))

# Sabitler
YH = 0.67
fE = 0.08

# ==========================================
# 2. MODEL FONKSİYONLARI
# ==========================================
def asm_model(y, t, params):
    Ss, Xs, Xh = y
    bh, kh, Ks, Kx, mumax = params
    
    # 1/gün -> 1/saat
    bh_h = bh / 24.0
    kh_h = kh / 24.0
    mumax_h = mumax / 24.0
    
    # Kinetik
    gro = mumax_h * (Ss / (Ks + Ss + 1e-12)) * Xh
    dec = bh_h * Xh
    
    if Xh > 1e-6: ratio = Xs / Xh
    else: ratio = 0
    
    hyd = kh_h * (ratio / (Kx + ratio + 1e-12)) * Xh
    
    dSs = - (1.0/YH)*gro + hyd
    dXs = - hyd
    dXh = gro - dec
    return [dSs, dXs, dXh]

def calculate_segment_our(params_kinetic, y0, time_segment):
    # Model sadece belirli bir zaman dilimi için çalıştırılır
    # Ancak başlangıç koşulları her zaman t=0'dan gelmelidir, 
    # bu yüzden simülasyonu t=0'dan başlatıp ilgili kısmı kesip alacağız.
    
    t_full = np.linspace(0, 1.0, 60) # Orijinal zaman adımları
    
    try:
        sol = odeint(asm_model, y0, t_full, args=(params_kinetic,), hmax=0.002)
    except:
        return np.ones(len(time_segment)) * 1e6
    
    # İlgili zaman dilimindeki indisleri bul
    mask = np.isin(t_full, time_segment)
    
    Ss = sol[mask, 0]
    Xs = sol[mask, 1]
    Xh = sol[mask, 2]
    
    bh, kh, Ks, Kx, mumax = params_kinetic
    
    Ss = np.maximum(Ss, 0)
    gro = (mumax/24.0) * (Ss / (Ks + Ss + 1e-12)) * Xh
    dec = (bh/24.0) * Xh
    
    our_segment = ((1.0 - YH)/YH)*gro + (1.0 - fE)*dec
    return our_segment

# ==========================================
# 3. PARÇA PARÇA (ARDIL) OPTİMİZASYON
# ==========================================
# Parametre Sözlüğü (Başlangıç Değerleri)
# Bu değerler her aşamada güncellenecek
curr_params = {
    'bh': 0.05, 'kh': 2.0, 'Ks': 5.0, 'Kx': 0.5, 'mumax': 4.0,
    'Ss0': 50.0, 'Xs0': 800.0, 'Xh0': 2000.0
}

print("--- FAZ 1: KUYRUK (Decay - bH) ---")
# Hedef: t > 0.8 (Son 12 nokta)
mask_tail = t_meas > 0.8
t_tail = t_meas[mask_tail]
data_tail = our_data_real[mask_tail]

def obj_phase1(x):
    # Sadece bh değişiyor
    p = [x[0], curr_params['kh'], curr_params['Ks'], curr_params['Kx'], curr_params['mumax']]
    y0 = [curr_params['Ss0'], curr_params['Xs0'], curr_params['Xh0']]
    model = calculate_segment_our(p, y0, t_tail)
    return np.sum((data_tail - model)**2)

res1 = minimize(obj_phase1, [0.1], bounds=[(0.01, 1.0)], method='L-BFGS-B')
curr_params['bh'] = res1.x[0]
print(f"Bulunan bH: {curr_params['bh']:.4f}")


print("\n--- FAZ 2: PİK (Growth - mumax, Ks, Ss0, Xh0) ---")
# Hedef: t < 0.15 (İlk 10 nokta)
mask_peak = t_meas < 0.15
t_peak = t_meas[mask_peak]
data_peak = our_data_real[mask_peak]

def obj_phase2(x):
    # x: [mumax, Ks, Ss0, Xh0]
    # bh sabit (Faz 1'den geldi)
    p = [curr_params['bh'], curr_params['kh'], x[1], curr_params['Kx'], x[0]]
    y0 = [x[2], curr_params['Xs0'], x[3]]
    model = calculate_segment_our(p, y0, t_peak)
    return np.sum((data_peak - model)**2)

# Bounds: mumax, Ks, Ss0, Xh0
bnds2 = [(1.0, 10.0), (0.1, 5.0), (10.0, 200.0), (1000.0, 4000.0)]
res2 = minimize(obj_phase2, [4.0, 1.0, 50.0, 2000.0], bounds=bnds2, method='L-BFGS-B')
curr_params['mumax'] = res2.x[0]
curr_params['Ks']    = res2.x[1]
curr_params['Ss0']   = res2.x[2]
curr_params['Xh0']   = res2.x[3]
print(f"Bulunan Pik Parametreleri: mumax={res2.x[0]:.2f}, Ks={res2.x[1]:.2f}, Ss0={res2.x[2]:.2f}, Xh0={res2.x[3]:.2f}")


print("\n--- FAZ 3: OMUZ (Hydrolysis - kh, Kx, Xs0) ---")
# Hedef: 0.15 < t < 0.6
mask_mid = (t_meas >= 0.15) & (t_meas <= 0.6)
t_mid = t_meas[mask_mid]
data_mid = our_data_real[mask_mid]

def obj_phase3(x):
    # x: [kh, Kx, Xs0]
    # Diğerleri sabit
    p = [curr_params['bh'], x[0], curr_params['Ks'], x[1], curr_params['mumax']]
    y0 = [curr_params['Ss0'], x[2], curr_params['Xh0']]
    model = calculate_segment_our(p, y0, t_mid)
    return np.sum((data_mid - model)**2)

# Bounds: kh, Kx, Xs0
bnds3 = [(0.1, 10.0), (0.01, 2.0), (100.0, 3000.0)]
res3 = minimize(obj_phase3, [2.0, 0.5, 800.0], bounds=bnds3, method='L-BFGS-B')
curr_params['kh']  = res3.x[0]
curr_params['Kx']  = res3.x[1]
curr_params['Xs0'] = res3.x[2]
print(f"Bulunan Omuz Parametreleri: kh={res3.x[0]:.2f}, Kx={res3.x[1]:.2f}, Xs0={res3.x[2]:.2f}")


print("\n--- FAZ 4: GLOBAL CİLA (Final Polish) ---")
# Artık elimizde çok iyi bir başlangıç noktası var.
# Tüm veriyi kullanarak çok dar bir aralıkta son optimizasyonu yapıyoruz.

def obj_global(x):
    # x: [bh, kh, Ks, Kx, mumax, Ss0, Xs0, Xh0]
    p = x[:5]
    y0 = x[5:]
    
    # Tüm zamanlar için
    t_full = np.linspace(0, 1.0, 60)
    try:
        sol = odeint(asm_model, y0, t_full, args=(p,), hmax=0.002)
        Ss, Xs, Xh = sol[:,0], sol[:,1], sol[:,2]
        Ss = np.maximum(Ss, 0)
        gro = (p[4]/24.0) * (Ss / (p[2] + Ss + 1e-12)) * Xh
        dec = (p[0]/24.0) * Xh
        model = ((1.0 - YH)/YH)*gro + (1.0 - fE)*dec
        sse = np.sum((our_data_real - model)**2)
        return sse
    except:
        return 1e9

# Başlangıç noktası = Ardışık yöntemden gelenler
p_init = [
    curr_params['bh'], curr_params['kh'], curr_params['Ks'], curr_params['Kx'], 
    curr_params['mumax'], curr_params['Ss0'], curr_params['Xs0'], curr_params['Xh0']
]

# Bounds'u bulduğumuz değerlerin %50 altı ve üstü olarak daraltalım ki optimizer sapıtmasın
bounds_global = []
for val in p_init:
    low = val * 0.5
    high = val * 1.5
    # Mantıksız sınırları düzelt
    if low < 0.001: low = 0.001
    bounds_global.append((low, high))

res_final = minimize(obj_global, p_init, method='L-BFGS-B', bounds=bounds_global, tol=1e-10)

print(f"\n===== FINAL SSE: {res_final.fun:.4f} =====")
labels = ["bh", "kh", "Ks", "Kx", "mumax", "Ss0", "Xs0", "Xh0"]
for l, v in zip(labels, res_final.x):
    print(f"{l}: {v:.4f}")

# Grafik Çiz
t_smooth = np.linspace(0, 1.0, 500)
sol = odeint(asm_model, res_final.x[5:], t_smooth, args=(res_final.x[:5],), hmax=0.002)
Ss, Xs, Xh = sol[:,0], sol[:,1], sol[:,2]
Ss = np.maximum(Ss, 0)
gro = (res_final.x[4]/24.0) * (Ss / (res_final.x[2] + Ss + 1e-12)) * Xh
dec = (res_final.x[0]/24.0) * Xh
our_smooth = ((1.0 - YH)/YH)*gro + (1.0 - fE)*dec

plt.figure(figsize=(10, 6))
plt.scatter(t_meas, our_data_real, facecolors='none', edgecolors='blue', s=80, label='Measurement', zorder=2)
plt.plot(t_smooth, our_smooth, 'r-', linewidth=2.5, label=f'Piecewise Optimized (SSE: {res_final.fun:.1f})', zorder=1)
plt.title('Sequential (Piecewise) Optimization Result')
plt.xlabel('Time (h)')
plt.ylabel('OUR (mgO2/l/h)')
plt.legend()
plt.grid(True)
plt.savefig('sequential_fit.png')
plt.show()