import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings

# Gereksiz uyarıları kapat
warnings.filterwarnings("ignore")

# ==========================================
# 1. VERİLER VE SABİTLER
# ==========================================
our_data_real = np.array([
    54.0, 55.2, 56.0, 55.6, 44.6, 32.0, 29.0, 26.0, 25.0, 25.5, 25.0, 25.2,
    25.1, 24.2, 24.7, 24.0, 23.7, 23.8, 23.0, 22.5, 22.7, 21.5, 20.6, 19.6,
    19.6, 19.5, 18.1, 17.5, 15.6, 15.2, 13.2, 12.0, 10.0, 9.1, 8.3, 6.6,
    6.1, 5.5, 5.5, 5.2, 5.1, 5.0, 4.6, 4.9, 5.1, 4.4, 4.5, 4.4,
    4.3, 4.3, 4.3, 4.2, 4.1, 4.0, 3.9, 3.9, 3.8, 3.8, 3.7, 3.6
])
t_meas = np.linspace(0, 1.0, len(our_data_real))

YH = 0.67
fE = 0.08

# ==========================================
# 2. MODEL FONKSİYONLARI
# ==========================================
def asm_model(y, t, params):
    Ss, Xs, Xh = y
    bh, kh, Ks, Kx, mumax = params
    
    # Process Rates (1/gün -> 1/saat)
    gro = (mumax/24.0) * (Ss / (Ks + Ss + 1e-12)) * Xh
    dec = (bh/24.0) * Xh
    
    ratio = Xs / (Xh + 1e-9) if Xh > 1e-6 else 0
    hyd = (kh/24.0) * (ratio / (Kx + ratio + 1e-12)) * Xh
    
    dSs = - (1.0/YH)*gro + hyd
    dXs = - hyd
    dXh = gro - dec
    return [dSs, dXs, dXh]

def calculate_our(p, y0, t_eval):
    try:
        # Hata toleranslarını artırarak (rtol, atol) hassas çözüm alıyoruz
        sol = odeint(asm_model, y0, t_eval, args=(p,), hmax=0.002, rtol=1e-9, atol=1e-9)
        Ss, Xs, Xh = sol[:,0], sol[:,1], sol[:,2]
        
        Ss = np.maximum(Ss, 0)
        gro = (p[4]/24.0) * (Ss / (p[2] + Ss + 1e-12)) * Xh
        dec = (p[0]/24.0) * Xh
        
        return ((1.0 - YH)/YH)*gro + (1.0 - fE)*dec
    except:
        return np.ones_like(t_eval) * 1e6

# ==========================================
# 3. ARDIŞIK (SEQUENTIAL) OPTİMİZASYON
# ==========================================
print("--- Bölge Bazlı Optimizasyon Başlıyor ---")
curr_params = {'bh': 0.05, 'kh': 2.0, 'Ks': 1.0, 'Kx': 0.5, 'mumax': 4.0, 'Ss0': 50.0, 'Xs0': 800.0, 'Xh0': 2000.0}

# FAZ 1: KUYRUK (t > 0.8) -> Sadece bh
def obj_tail(x):
    p = [x[0], curr_params['kh'], curr_params['Ks'], curr_params['Kx'], curr_params['mumax']]
    y0 = [curr_params['Ss0'], curr_params['Xs0'], curr_params['Xh0']]
    our = calculate_our(p, y0, t_meas)
    return np.sum((our_data_real[45:] - our[45:])**2)

res1 = minimize(obj_tail, [0.05], bounds=[(0.001, 1.0)], method='L-BFGS-B')
curr_params['bh'] = res1.x[0]

# FAZ 2: PİK (t < 0.15) -> mumax, Ks, Ss0, Xh0
def obj_peak(x):
    p = [curr_params['bh'], curr_params['kh'], x[1], curr_params['Kx'], x[0]]
    y0 = [x[2], curr_params['Xs0'], x[3]]
    our = calculate_our(p, y0, t_meas)
    return np.sum((our_data_real[:15] - our[:15])**2)

bnds2 = [(1.0, 10.0), (0.1, 5.0), (10.0, 200.0), (1000.0, 4000.0)]
res2 = minimize(obj_peak, [4.0, 1.0, 40.0, 2000.0], bounds=bnds2, method='L-BFGS-B')
curr_params.update({'mumax': res2.x[0], 'Ks': res2.x[1], 'Ss0': res2.x[2], 'Xh0': res2.x[3]})

# FAZ 3: OMUZ (0.15 < t < 0.6) -> kh, Kx, Xs0
def obj_shoulder(x):
    p = [curr_params['bh'], x[0], curr_params['Ks'], x[1], curr_params['mumax']]
    y0 = [curr_params['Ss0'], x[2], curr_params['Xh0']]
    our = calculate_our(p, y0, t_meas)
    return np.sum((our_data_real[15:45] - our[15:45])**2)

bnds3 = [(0.1, 10.0), (0.01, 2.0), (100.0, 3000.0)]
res3 = minimize(obj_shoulder, [2.0, 0.5, 800.0], bounds=bnds3, method='L-BFGS-B')
curr_params.update({'kh': res3.x[0], 'Kx': res3.x[1], 'Xs0': res3.x[2]})

print("Bölge Bazlı Tahmin Tamamlandı.")

# ==========================================
# 4. MULTI-START GLOBAL POLISH
# ==========================================
print("--- Çoklu Başlangıçlı Global Cila (Polish) Başlıyor ---")

def obj_global(x):
    p = x[:5]
    y0 = x[5:]
    our = calculate_our(p, y0, t_meas)
    return np.sum((our_data_real - our)**2)

# Temel aday (Bölge bazlı sonuçlar)
p_base = [curr_params[k] for k in ['bh', 'kh', 'Ks', 'Kx', 'mumax', 'Ss0', 'Xs0', 'Xh0']]

# 5 farklı pertürbasyon (küçük sapmalı adaylar) oluşturuyoruz
candidates = [p_base]
for _ in range(5):
    # %10 sapmalı rastgele adaylar
    candidates.append([val * np.random.uniform(0.9, 1.1) for val in p_base])

best_sse = 1e9
best_params = []

for i, start_point in enumerate(candidates):
    # Sınırları başlangıç noktasının etrafında daralt (Local Search)
    bnds_global = []
    for val in start_point:
        bnds_global.append((val*0.5, val*1.5))
    
    # Fiziksel sınır düzeltmeleri
    bnds_global[0] = (0.001, 1.0) # bh
    bnds_global[2] = (0.1, 5.0)   # Ks (Düşük tutuyoruz)
    
    res = minimize(obj_global, start_point, bounds=bnds_global, method='L-BFGS-B', tol=1e-11)
    
    if res.fun < best_sse:
        best_sse = res.fun
        best_params = res.x
        print(f"Yeni En İyi SSE: {best_sse:.4f} (Aday {i+1})")

# ==========================================
# 5. SONUÇLAR VE GRAFİK
# ==========================================
print(f"\n===== NİHAİ PARAMETRELER =====")
print(f"En İyi SSE: {best_sse:.4f}")
labels = ["bh", "kh", "Ks", "Kx", "mumax", "Ss0", "Xs0", "Xh0"]
for l, v in zip(labels, best_params):
    print(f"{l}: {v:.4f}")

# Grafik Çizimi
t_smooth = np.linspace(0, 1.0, 500)
p_final_kinetic = best_params[:5]
y0_final = best_params[5:]
our_smooth = calculate_our(p_final_kinetic, y0_final, t_smooth)

plt.figure(figsize=(10, 6))
plt.scatter(t_meas, our_data_real, facecolors='none', edgecolors='blue', s=80, label='Ölçüm', zorder=2)
plt.plot(t_smooth, our_smooth, 'r-', linewidth=2.5, label=f'Model Fit (SSE: {best_sse:.1f})', zorder=1)
plt.title('Final Optimized Model Fit')
plt.xlabel('Time (h)')
plt.ylabel('OUR (mgO2/l/h)')
plt.legend()
plt.grid(True)
plt.savefig('final_best_fit.png')
plt.show()