import cv2
import numpy as np
import pytesseract
import os
import csv

pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'

per = 30

# --- WZOR dowodu rejestracyjnego /// zczytanie zdjecia  --- #
# Load the image
img = cv2.imread("wzor.png")
# Up-sample

# Convert to the gray-scale
gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Simple-threshold
imgQ = cv2.threshold(gry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imshow("WZOR",imgQ)

# --- rysowanie punktow  --- #
orb = cv2.ORB_create(3000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

# --- SKAN dowodu rejestracyjnego /// zczytanie zdjecia  --- #
img = cv2.imread("wzor.png")
# Up-sample

# Convert to the gray-scale
gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Simple-threshold
imgOpel = cv2.threshold(gry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# --- rysowanie punktow  --- #
orb = cv2.ORB_create(2000)
kp2, des2 = orb.detectAndCompute(imgOpel,None)
impKp2 = cv2.drawKeypoints(imgOpel,kp2,None)

# --- MORPHING  // porownianie skanow  --- #
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(des2,des1)
matches.sort(key=lambda x: x.distance)
good = matches[:int(len(matches)*(per/100))]
imgMatch = cv2.drawMatches(imgOpel,kp2,imgQ,kp1,good,None,flags=2)

cv2.imshow("Results",imgMatch)

srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
desPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

M, _=   cv2.findHomography(srcPoints,desPoints,cv2.RANSAC,5.0)
imgScan = cv2.warpPerspective(imgOpel,M,(882,1397))

cv2.imshow("XXX",imgScan)
imgShow = imgQ.copy()
imgMask = np.zeros_like(imgShow)

#A. Nr rejestracyjny
A_start_point = (130, 68)
A_end_point = (280, 90)
#B. Data pierwszej rejestracji
B_start_point = (310, 70)
B_end_point = (450, 90)
#C1. Nazwisko i imie wlasciciela, lub nazwa osoby prywatnej
C1_start_point = (139, 90)
C1_end_point = (440, 190)
#C3. Adres siedziby wlasciciela
C3_start_point = (139, 274)
C3_end_point = (755, 400)
#D1. Marka
D1_start_point = (140, 396)
D1_end_point = (360, 420)
#D2. Model, typ pojazdu
D2_start_point = (140, 422)
D2_end_point = (360, 445)
#D.2.1. Kod identyfikacyjny CNIT (ma znaczenie dla importu pojkazdu z innego kraju niz UE)
D21_start_point = (530, 445)
D21_end_point = (775, 470)
#D3. Nazwa handlowa
D3_start_point = (140, 480)
D3_end_point = (400, 500)
#E. Numer identyfikacyjny VIN
E_start_point = (545, 480)
E_end_point = (775, 503)
#F1. Masa max
F1_start_point = (140, 500)
F1_end_point = (260, 522)
#F2. Masa max
F2_start_point = (310, 500)
F2_end_point = (439, 522)
#F3. Masa max
F3_start_point = (475, 500)
F3_end_point = (600, 522)
#G. Masa max
G_start_point = (140, 525)
G_end_point = (260, 548)
#G1. Masa max
G1_start_point = (308, 525)
G1_end_point = (430, 548)
#J.
J_start_point = (140, 547)
J_end_point = (260, 570)
#J1
J1_start_point = (308, 547)
J1_end_point = (430, 570)
#J2
J2_start_point = (470, 547)
J2_end_point = (560, 570)
#J3
J3_start_point = (595, 547)
J3_end_point = (750, 570)
#K Nr recepcji welug typu (import)
K_start_point = (140,570)
K_end_point = (430,590)
#P1
P1_start_point = (140,592)
P1_end_point = (260,612)
#P2
P2_start_point = (305,592)
P2_end_point = (430,612)
#P3
P3_start_point = (473,592)
P3_end_point = (560,612)
#P6
P6_start_point = (600,592)
P6_end_point = (700,612)
#Q
Q_start_point = (140,612)
Q_end_point = (260,635)
#S1
S1_start_point = (305,612)
S1_end_point = (430,635)
#S2
S2_start_point = (473,612)
S2_end_point = (560,635)
#U1
U1_start_point = (600,612)
U1_end_point = (700,635)
#U2
U2_start_point = (140,640)
U2_end_point = (260,660)
#V7
V7_start_point = (305,640)
V7_end_point = (425,660)
#V9
V9_start_point = (470,640)
V9_end_point = (570,660)

color = (255, 0, 0)
thickness = 2

cv2.rectangle(imgMask, A_start_point, A_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, B_start_point, B_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, C1_start_point, C1_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, C3_start_point, C3_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, D1_start_point, D1_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, D2_start_point, D2_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, D21_start_point, D21_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, D3_start_point, D3_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, E_start_point, E_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, F1_start_point, F1_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, F2_start_point, F2_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, F3_start_point, F3_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, G_start_point, G_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, G1_start_point, G1_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, J_start_point, J_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, J1_start_point, J1_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, J2_start_point, J2_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, J3_start_point, J3_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, K_start_point, K_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, P1_start_point, P1_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, P2_start_point, P2_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, P3_start_point, P3_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, P6_start_point, P6_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, Q_start_point, Q_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, S1_start_point, S1_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, S2_start_point, S2_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, U1_start_point, U1_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, U2_start_point, U2_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, V7_start_point, V7_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, V9_start_point, V9_end_point, color, cv2.FILLED)

imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.9,0)

cv2.imshow("MASKA", imgMask)
cv2.imshow("Oryginal_maska", imgShow)

imgCrop1 = imgScan[70:90, 130:280]
cv2.imshow("A. - nr rejestracyjny", imgCrop1)

imgCrop2 = imgScan[70:93, 310:450]
cv2.imshow("B - Data pierwszej rejestracji", imgCrop2)

imgCrop3 = imgScan[90:195, 140:400]
cv2.imshow("C.1 - Wlasciciel pojazdu", imgCrop3)

imgCrop4 = imgScan[273:395, 140:470]
cv2.imshow("C.3 - Adres wlasciciela", imgCrop4)

imgCrop5 = imgScan[395:420, 139:470]
cv2.imshow("D.1 - Marka", imgCrop5)

imgCrop6 = imgScan[420:445, 140:430]
cv2.imshow("D.2 - Typ pojazdu", imgCrop6)

imgCrop7 = imgScan[440:470, 530:730]
cv2.imshow("D.2.1 - Kod identyfikacyjny CNIT", imgCrop7)

imgCropE = imgScan[477:503, 545:770]
cv2.imshow("E - nr identyfikacyjny VIN", imgCropE)
imgCrop8 = imgScan[480:500, 140:460]
cv2.imshow("D.3 - Nazwa Handlowa", imgCrop8)
imgCrop9 = imgScan[500:523, 140:260]
cv2.imshow("F1 - Masa max calkowita pojazdu z ladunkiem", imgCrop9)
imgCrop10 = imgScan[500:523, 305:430]
cv2.imshow("F2 - Masa max calkowita pojazdu dop we FR", imgCrop10)
imgCrop11 = imgScan[500:523, 475:580]
cv2.imshow("F3 - Masa max calkowita pojazdu dop we FR z przyczepa", imgCrop11)
imgCrop12 = imgScan[525:550, 140:260]
cv2.imshow("G - masa pojazdu do jazdy z kierowca bez ładunku (na pusto)", imgCrop12)
imgCrop13 = imgScan[525:550, 308:430]
cv2.imshow("G1 - masa pustego pojazdu (bez ładunku, bez kierowcy)", imgCrop13)
imgCrop14 = imgScan[547:575, 140:185]
cv2.imshow("J - Kategoria pojazdu", imgCrop14)
imgCrop15 = imgScan[547:572, 305:430]
cv2.imshow("J.1 - Rodzaj pojazdu w/g oznaczen Francuskich", imgCrop15)
imgCrop16 = imgScan[547:570, 470:560]
cv2.imshow("J.2 - Typ karoserii w/g norm Europejskich", imgCrop16)
imgCrop17 = imgScan[547:570, 600:750]
cv2.imshow("J.3 - Typ karoserii w/g norm Francuskich", imgCrop17)
imgCrop18 = imgScan[570:592, 140:440]
cv2.imshow("K - nr recepcji w/g typu (dot. importu z dowolnego kraju)", imgCrop18)
imgCrop19 = imgScan[590:619, 140:260]
cv2.imshow("P.1 - pojemność silnika w cm3", imgCrop19)
imgCrop20 = imgScan[590:619, 305:430]
cv2.imshow("P.2 - moc silnika w kW", imgCrop20)
imgCrop21 = imgScan[590:619, 475:560]
cv2.imshow("P.3 - Rodzaj paliwa (GO - disel, ES - benzyna)", imgCrop21)
imgCrop22 = imgScan[590:619, 600:720]
cv2.imshow("P.6 - Moc administracyjna we Francji", imgCrop22)
imgCrop23 = imgScan[612:640, 140:260]
cv2.imshow("Q - (tylko dla motocykli)", imgCrop23)
imgCrop24 = imgScan[612:640, 305:345]
cv2.imshow("S.1 - ilość miejsc siedzących (łacznie z kierowca)", imgCrop24)
imgCrop25 = imgScan[612:640, 475:560]
cv2.imshow("S.2 - ilość miejsc stojących", imgCrop25)
imgCrop26 = imgScan[612:639, 600:680]
cv2.imshow("U.1 - poziom halasu silnika w [dB]", imgCrop26)

imgCrop27 = imgScan[640:664, 140:260]
cv2.imshow("U.2 - Prędkość obrotowa silnika [obr/min]", imgCrop27)
imgCrop28 = imgScan[640:664, 305:420]
cv2.imshow("V.7 - Emisja CO2 [g/km]", imgCrop28)
imgCrop29 = imgScan[640:664, 470:740]
cv2.imshow("V.9 - spełniana forma emisji spalin w/g UE", imgCrop29)

imgCrop30 = imgScan[660:680, 330:470]
cv2.imshow("X.1 - Data ważności przeglądu technicznego", imgCrop30)

imgCrop31 = imgScan[560:580, 204:280]
cv2.imshow("Y.1 - Moc administracyjna we Francji", imgCrop31)
imgCrop32 = imgScan[560:580, 322:380]
cv2.imshow("Y.2 - Moc administracyjna we Francji", imgCrop32)
imgCrop33 = imgScan[577:593, 203:280]
cv2.imshow("Y.3 - Moc administracyjna we Francji", imgCrop33)
imgCrop34 = imgScan[577:593, 322:380]
cv2.imshow("Y.4 - Moc administracyjna we Francji", imgCrop34)
imgCrop35 = imgScan[590:610, 202:280]
cv2.imshow("Y.5 - Moc administracyjna we Francji", imgCrop35)
imgCrop36 = imgScan[590:610, 323:380]
cv2.imshow("Y.6 - Moc administracyjna we Francji", imgCrop36)

imgCrop37 = imgScan[798:820, 140:280]
cv2.imshow("I - Data wystawienia dowodu", imgCrop37)

imgCrop333 = imgScan[1066:1270, 110:450]
cv2.imshow("Dolna czesc - odcieta", imgCrop333)
imgCrop444 = imgScan[1290:1370, 100:750]
cv2.imshow("Kod na dole", imgCrop444)



TGREEN =  '\033[32m' # Green Text
ENDC = '\033[m' # reset to the defaults

from datetime import date
today = date.today()
d1 = today.strftime("%d/%m/%Y")
print("OTMUCHÓW", d1, '\n')

print('UWIERZYTELNIONE TŁUMACZENIE Z JĘZYKA FRANCUSKIEGO - DOWÓD REJESTRACYJNY POJAZDU \n')
print('REPUBLIKA FRANCUSKA  F Unia Europejska   Ministerstwo Spraw Wewnętrznych \n')
print('Dowód rejestracyjny fr - dane z cz_1: \n')
print('(A.) Nr rejestracyjny:', TGREEN + pytesseract.image_to_string(imgCrop1), ENDC)
print('(B) Data pierwszej rejestracji:', TGREEN + pytesseract.image_to_string(imgCrop2), ENDC)
print('(C.1) Własciciel:', TGREEN + pytesseract.image_to_string(imgCrop3), ENDC)
print('(C.3) Adres:', TGREEN + pytesseract.image_to_string(imgCrop4), ENDC)
print('(D.1) Marka:', TGREEN + pytesseract.image_to_string(imgCrop5), ENDC)
print('(D.2) Model:', TGREEN + pytesseract.image_to_string(imgCrop6), ENDC)
print('(D.2.1) Kod identyfikacyjny CNIT:', TGREEN + pytesseract.image_to_string(imgCrop7), ENDC)
print('(E) Nr VIN:',  TGREEN +pytesseract.image_to_string(imgCropE), ENDC)
print('(D.3) Nazwa Handlowa:', TGREEN + pytesseract.image_to_string(imgCrop8), ENDC)
print('(F1) Masa max calkowita pojazdu z ladunkiem:', TGREEN + pytesseract.image_to_string(imgCrop9), ENDC)
print('(F2) Masa max calkowita pojazdu dop we FR:', TGREEN + pytesseract.image_to_string(imgCrop10), ENDC)
print('(F3) Masa max calkowita pojazdu dop we FR z przyczepa:', TGREEN + pytesseract.image_to_string(imgCrop11), ENDC)
print('(G) Masa pojazdu do jazdy z kierowca:', TGREEN + pytesseract.image_to_string(imgCrop12), ENDC)
print('(G.1) Masa pustego pojazdu:', TGREEN + pytesseract.image_to_string(imgCrop13), ENDC)
print('(J) Kategoria pojazdu:', TGREEN + pytesseract.image_to_string(imgCrop14), ENDC)
print('(J.1) Rodzaj pojazdu w/g oznaczen Francuskich:', TGREEN + pytesseract.image_to_string(imgCrop15), ENDC)
print('(J.2) Typ karoserii w/g norm Europejskich:', TGREEN + pytesseract.image_to_string(imgCrop16), ENDC)
print('(J.3) Typ karoserii w/g norm Francuskich:', TGREEN + pytesseract.image_to_string(imgCrop17), ENDC)
print('(K) Nr recepcji w/g typu (dot. importu z dowolnego kraju):', TGREEN + pytesseract.image_to_string(imgCrop18), ENDC)
print('(P.1) Pojemność silnika w cm3:', TGREEN + pytesseract.image_to_string(imgCrop19), ENDC)
print('(P.2) Moc silnika w kW:', TGREEN + pytesseract.image_to_string(imgCrop20), ENDC)
print('(P.3) Rodzaj paliwa (GO - disel, ES - benzyna):', TGREEN + pytesseract.image_to_string(imgCrop21), ENDC)
print('(P.6) Moc administracyjna we Francji:', TGREEN + pytesseract.image_to_string(imgCrop22), ENDC)
print('(Q) (tylko dla motocykli):', TGREEN + pytesseract.image_to_string(imgCrop23), ENDC)
print('(S.1) Ilość miejsc siedzących (łacznie z kierowca):', TGREEN + pytesseract.image_to_string(imgCrop24), ENDC)
print('(S.2) Ilość miejsc stojących:', TGREEN + pytesseract.image_to_string(imgCrop25), ENDC)
print('(U.1) Poziom halasu silnika w [dB]:', TGREEN + pytesseract.image_to_string(imgCrop26), ENDC)
print('(U.2) Prędkość obrotowa silnika [obr/min]:', TGREEN + pytesseract.image_to_string(imgCrop27), ENDC)
print('(V.7) Emisja CO2 [g/km]:', TGREEN + pytesseract.image_to_string(imgCrop28), ENDC)
print('(V.9) Spełniana forma emisji spalin w/g UE:', TGREEN + pytesseract.image_to_string(imgCrop29), ENDC)
print('(X.1) Data ważności przeglądu technicznego:', TGREEN + pytesseract.image_to_string(imgCrop30), ENDC)

print('(I) Data wystawienia dowodu:', TGREEN + pytesseract.image_to_string(imgCrop37), ENDC)
print('Dolna czesc odcieta:\n',TGREEN+pytesseract.image_to_string(imgCrop333),ENDC)
print('Kod na dole:\n',TGREEN+pytesseract.image_to_string(imgCrop444),ENDC)


tex_1 = pytesseract.image_to_string(imgCrop1)
x_tex_1 = tex_1.rstrip()
tex_2 = pytesseract.image_to_string(imgCrop2)
x_tex_2 = tex_2.rstrip()
tex_3 = pytesseract.image_to_string(imgCrop3)
x_tex_3 = tex_3.rstrip()
tex_4 = pytesseract.image_to_string(imgCrop4)
x_tex_4 = tex_4.rstrip()

with open('DANE_auta2.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file)

    employee_writer.writerow(['ID', 'Nr rejestracyjny', 'Data rejestracji', 'Właściciel pojazdu', 'Adres  wlasciciela'])
    employee_writer.writerow(['1', x_tex_1, x_tex_2, x_tex_3, x_tex_4])



# --- KONIEC  --- #
cv2.waitKey(0)