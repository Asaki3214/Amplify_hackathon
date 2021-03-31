import numpy as np
import csv
from collections import defaultdict
import wave
import struct

from amplify import gen_symbols, BinaryPoly, sum_poly, Solver, decode_solution
from amplify.client import FixstarsClient

num_bar = 2 #作曲される曲の小節数
num_length = 16*num_bar #16分音符の個数に変換
param_constraints = 100 #OneHotEncoding項の強さ
param_length_constraints = 100 #曲の小節数に対する制約項の強さ

#音符一覧
notes_array = ['C4_0', 'D4_0', 'E4_0', 'F4_0', 'G4_0', 'A4_0', 'B4_0', 'C5_0', 'D5_0', 'E5_0', 'F5_0', 'G5_0', 'A5_0', 'B5_0', 'qq_0', 
         'C4_1', 'D4_1', 'E4_1', 'F4_1', 'G4_1', 'A4_1', 'B4_1', 'C5_1', 'D5_1', 'E5_1', 'F5_1', 'G5_1', 'A5_1', 'B5_1', 'qq_1', 
         'C4_2', 'D4_2', 'E4_2', 'F4_2', 'G4_2', 'A4_2', 'B4_2', 'C5_2', 'D5_2', 'E5_2', 'F5_2', 'G5_2', 'A5_2', 'B5_2', 'qq_2', 
         'C4_3', 'D4_3', 'E4_3', 'F4_3', 'G4_3', 'A4_3', 'B4_3', 'C5_3', 'D5_3', 'E5_3', 'F5_3', 'G5_3', 'A5_3', 'B5_3', 'qq_3', 
         'C4_5', 'D4_5', 'E4_5', 'F4_5', 'G4_5', 'A4_5', 'B4_5', 'C5_5', 'D5_5', 'E5_5', 'F5_5', 'G5_5', 'A5_5', 'B5_5', 'qq_5',
         'C4_7', 'D4_7', 'E4_7', 'F4_7', 'G4_7', 'A4_7', 'B4_7', 'C5_7', 'D5_7', 'E5_7', 'F5_7', 'G5_7', 'A5_7', 'B5_7', 'qq_7', 'null']
notes = len(notes_array) 

#楽曲データの読み込み
with open('music_data.csv',newline="", encoding='shift_jis') as f:
    reader = csv.reader(f)
    teacher = [[note for note in row if note != ''] for row in reader]

#音符を数値に変換
for music in teacher: 
    for i in range(len(music)-1):
        music[i+1] = notes_array.index(music[i+1])

#楽曲データの音符遷移を評価
def weight_maker(teacher):
    kanon_matrix = defaultdict(float)
    for music in teacher:
        for i in range(len(music)-2):
            trans = music[i+1]//15 + music[i+2]//15 #最大値10
            kanon_matrix[(music[i+1],music[i+2])] += trans/10    
    return kanon_matrix

kanon_matrix = weight_maker(teacher)

#変数間の重み行列を作成
weights = [[kanon_matrix[qx,qy] for qy in range(notes)] for qx in range(notes)]


#コスト関数
q = gen_symbols(BinaryPoly, num_length, notes)
constraints = sum_poly(num_length, lambda n: (
                 sum_poly(notes, lambda i: q[n][i]) - 1) ** 2)

note_length = [1,2,3,4,6,8]
sum_notes = sum_poly(num_length,
                lambda n: sum_poly(len(note_length),
                    lambda k: sum_poly(15, 
                        lambda i :q[n][i+15*k]*note_length[k]
                    ),
                ),
            )
length_constraints = (sum_notes - num_length)**2

cost = sum_poly(
    num_length-1,
    lambda n: sum_poly(
        notes-1,
        lambda i: sum_poly(
            notes-1, lambda j: weights[i][j] * q[n][i] * q[(n + 1) % (notes-1)][j]
        ),
    ),
)

constraints *= param_constraints
length_constraints *= param_length_constraints

model = -cost + constraints + length_constraints

#QUBOをAmplifyで解く
client = FixstarsClient()
client.parameters.timeout = 5000
#client.token = "DELETE TOKEN"
solver = Solver(client)
result = solver.solve(model)
if len(result) == 0:
    print("Any one of constraints is not satisfied.")

#解をデコード
for sol in result:
    solution = decode_solution(q, sol.values)

codes = [notes_array[i] for n in range(num_length) for i in range(notes) if solution[n][i] == 1]

#作曲された譜面を表示(nullは読み飛ばす)
print(codes)

#各音符の周波数と長さを定義
VBA_Beep = {'C4':262, 'D4':294, 'E4':330, 'F4':349, 'G4':392, 'A4':440, 'B4':494, 'C5':523, 'D5':587, 'E5':659, 'F5': 698, 'G5':783, 'A5':880, 'B5':967, 'qq':0}
VBA_Beep_tempo = {'o16':0.25, 'o8':0.5, 'o6':0.75, 'o4':1, 'o3':1.5, 'o2':2}

#音源作成
tie_list = list(VBA_Beep_tempo.keys())
codes_melody = [note[0:2] for note in codes if note != 'null']
codes_tempo = [tie_list[int(note[3])] for note in codes if note != 'null']

Beep = [VBA_Beep[i] for i in codes_melody]
Beep_tempo = [VBA_Beep_tempo[i] for i in codes_tempo]

sec = 1 
sample_hz = 44100 
t= np.array([])
wv = np.array([])

for (note_hz, tp) in zip(Beep,Beep_tempo):
    t1 = np.arange(0, sample_hz * sec*tp)
    wv1 = np.sin(2 * np.pi * note_hz * t1/sample_hz) 
    t = np.append(t,t1)
    wv = np.append(wv,wv1)

#音源保存
max_num = 32767.0 / max(wv) 
wv16 = [int(x * max_num) for x in wv] 
bi_wv = struct.pack("h" * len(wv16), *wv16) 

with wave.open('music_01.wav', mode='wb') as file:
    param_wv = (1,2,sample_hz,len(bi_wv),'NONE','not compressed')
    file.setparams(param_wv)
    file.writeframes(bi_wv)
