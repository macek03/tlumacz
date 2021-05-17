import cv2
import numpy as np
import pytesseract
import os
import csv

pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'

per = 30

# --- WZOR dowodu rejestracyjnego /// zczytanie zdjecia  --- #
# Load the image
imgQ = cv2.imread("wzor.png")


# --- rysowanie punktow  --- #
orb = cv2.ORB_create(3000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

# --- SKAN dowodu rejestracyjnego /// zczytanie zdjecia  --- #
imgOpel = cv2.imread("wzor.png")


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
imgShow = imgOpel.copy()
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
#X1
X1_start_point = (330,657)
X1_end_point = (460,680)
#Y1
Y1_start_point = (140,680)
Y1_end_point = (260,700)
#Y2
Y2_start_point = (305,680)
Y2_end_point = (400,700)
#Y3
Y3_start_point = (140,700)
Y3_end_point = (260,722)
#Y4
Y4_start_point = (305,700)
Y4_end_point = (400,722)
#Y5
Y5_start_point = (140,722)
Y5_end_point = (305,750)
#Y6
Y6_start_point = (305,722)
Y6_end_point = (400,750)
#I
I_start_point = (140,795)
I_end_point = (280,820)
#Dane na dole
Cz_odc_start_point = (110,1070)
Cz_odc_end_point = (420,1270)
#KOD
KOD_start_point = (100,1290)
KOD_end_point = (780,1366)


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
cv2.rectangle(imgMask, X1_start_point, X1_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, Y1_start_point, Y1_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, Y2_start_point, Y2_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, Y3_start_point, Y3_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, Y4_start_point, Y4_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, Y5_start_point, Y5_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, Y6_start_point, Y6_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, I_start_point, I_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, Cz_odc_start_point, Cz_odc_end_point, color, cv2.FILLED)
cv2.rectangle(imgMask, KOD_start_point, KOD_end_point, color, cv2.FILLED)

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

imgCrop31 = imgScan[680:705, 140:266]
cv2.imshow("Y1 - Podatek regionalny", imgCrop31)
imgCrop32 = imgScan[680:705, 305:425]
cv2.imshow("Y2- Podatek dodatkowy", imgCrop32)
imgCrop33 = imgScan[700:724, 140:266]
cv2.imshow("Y3 - Podatek ekologiczny", imgCrop33)
imgCrop34 = imgScan[700:724, 305:425]
cv2.imshow("Y4 - Podatek za usluge administracyjna", imgCrop34)
imgCrop35 = imgScan[722:747, 140:266]
cv2.imshow("Y5 - Podatek za wysylke droga pocztowa", imgCrop35)
imgCrop36 = imgScan[722:747, 305:425]
cv2.imshow("Y6 - Suma podatku", imgCrop36)

imgCrop37 = imgScan[798:820, 140:280]
cv2.imshow("I - Data wystawienia dowodu", imgCrop37)

imgCrop333 = imgScan[1066:1270, 110:450]
cv2.imshow("Dolna czesc - odcieta", imgCrop333)
imgCrop444 = imgScan[1290:1370, 100:750]
cv2.imshow("Kod na dole", imgCrop444)

imgCrop555 = imgScan[1330:1355, 550:708]
cv2.imshow("Oznaczenie dokumentu", imgCrop555)



TGREEN =  '\033[32m' # Green Text
ENDC = '\033[m' # reset to the defaults

from datetime import date
today = date.today()
d1 = today.strftime("%d/%m/%Y")
print("OTMUCHÓW", TGREEN+d1,'\n',ENDC)

print('UWIERZYTELNIONE TŁUMACZENIE Z JĘZYKA FRANCUSKIEGO - DOWÓD REJESTRACYJNY POJAZDU \n')
print('REPUBLIKA FRANCUSKA  F Unia Europejska   Ministerstwo Spraw Wewnętrznych \n')
print('Oznaczenie dokumentu:',TGREEN+pytesseract.image_to_string(imgCrop555),ENDC)
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

print('(Y.1) Podatek regionalny:', TGREEN + pytesseract.image_to_string(imgCrop31), ENDC)
print('(Y.2) Podatek dodatkowy:', TGREEN + pytesseract.image_to_string(imgCrop32), ENDC)
print('(Y.3) Podatek ekologiczny:', TGREEN + pytesseract.image_to_string(imgCrop33), ENDC)
print('(Y.4) Podatek za usluge administracyjna:', TGREEN + pytesseract.image_to_string(imgCrop34), ENDC)
print('(Y.5) Podatek za wysylke droga pocztowa:', TGREEN + pytesseract.image_to_string(imgCrop35), ENDC)
print('(Y.6) Suma:', TGREEN + pytesseract.image_to_string(imgCrop36), ENDC)

print('(I) Data wystawienia dowodu:', TGREEN + pytesseract.image_to_string(imgCrop37), ENDC)
print('Dolna czesc odcieta:\n',TGREEN+pytesseract.image_to_string(imgCrop333),ENDC)
print('Kod na dole:\n',TGREEN+pytesseract.image_to_string(imgCrop444),ENDC)


tex_0 = pytesseract.image_to_string(imgCrop555)
x_tex_0 = tex_0.rstrip()
tex_1 = pytesseract.image_to_string(imgCrop1)
x_tex_1 = tex_1.rstrip()
tex_2 = pytesseract.image_to_string(imgCrop2)
x_tex_2 = tex_2.rstrip()
tex_3 = pytesseract.image_to_string(imgCrop3)
x_tex_3 = tex_3.rstrip()
tex_4 = pytesseract.image_to_string(imgCrop4)
x_tex_4 = tex_4.rstrip()
tex_5 = pytesseract.image_to_string(imgCrop5)
x_tex_5 = tex_5.rstrip()
tex_6 = pytesseract.image_to_string(imgCrop6)
x_tex_6 = tex_6.rstrip()
tex_7 = pytesseract.image_to_string(imgCrop7)
x_tex_7 = tex_7.rstrip()
tex_E = pytesseract.image_to_string(imgCropE)
x_tex_E = tex_E.rstrip()
tex_8 = pytesseract.image_to_string(imgCrop8)
x_tex_8 = tex_8.rstrip()
tex_9 = pytesseract.image_to_string(imgCrop9)
x_tex_9 = tex_9.rstrip()
tex_10 = pytesseract.image_to_string(imgCrop10)
x_tex_10 = tex_10.rstrip()
tex_11 = pytesseract.image_to_string(imgCrop11)
x_tex_11 = tex_11.rstrip()
tex_12 = pytesseract.image_to_string(imgCrop12)
x_tex_12 = tex_12.rstrip()
tex_13 = pytesseract.image_to_string(imgCrop13)
x_tex_13 = tex_13.rstrip()
tex_14 = pytesseract.image_to_string(imgCrop14)
x_tex_14 = tex_14.rstrip()
tex_15 = pytesseract.image_to_string(imgCrop15)
x_tex_15 = tex_15.rstrip()
tex_16 = pytesseract.image_to_string(imgCrop16)
x_tex_16 = tex_16.rstrip()
tex_17 = pytesseract.image_to_string(imgCrop17)
x_tex_17 = tex_17.rstrip()
tex_18 = pytesseract.image_to_string(imgCrop18)
x_tex_18 = tex_18.rstrip()
tex_19 = pytesseract.image_to_string(imgCrop19)
x_tex_19 = tex_19.rstrip()
tex_20 = pytesseract.image_to_string(imgCrop20)
x_tex_20 = tex_20.rstrip()
tex_21 = pytesseract.image_to_string(imgCrop21)
x_tex_21 = tex_21.rstrip()
tex_22 = pytesseract.image_to_string(imgCrop22)
x_tex_22 = tex_22.rstrip()
tex_23 = pytesseract.image_to_string(imgCrop23)
x_tex_23 = tex_23.rstrip()
tex_24 = pytesseract.image_to_string(imgCrop24)
x_tex_24 = tex_24.rstrip()
tex_25 = pytesseract.image_to_string(imgCrop25)
x_tex_25 = tex_25.rstrip()
tex_26 = pytesseract.image_to_string(imgCrop26)
x_tex_26 = tex_26.rstrip()
tex_27 = pytesseract.image_to_string(imgCrop27)
x_tex_27 = tex_27.rstrip()
tex_28 = pytesseract.image_to_string(imgCrop28)
x_tex_28 = tex_28.rstrip()
tex_29 = pytesseract.image_to_string(imgCrop29)
x_tex_29 = tex_29.rstrip()
tex_30 = pytesseract.image_to_string(imgCrop30)
x_tex_30 = tex_30.rstrip()
tex_31 = pytesseract.image_to_string(imgCrop31)
x_tex_31 = tex_31.rstrip()
tex_32 = pytesseract.image_to_string(imgCrop32)
x_tex_32 = tex_32.rstrip()
tex_33 = pytesseract.image_to_string(imgCrop33)
x_tex_33 = tex_33.rstrip()
tex_34 = pytesseract.image_to_string(imgCrop34)
x_tex_34 = tex_34.rstrip()
tex_35 = pytesseract.image_to_string(imgCrop35)
x_tex_35 = tex_35.rstrip()
tex_36 = pytesseract.image_to_string(imgCrop36)
x_tex_36 = tex_36.rstrip()
tex_37 = pytesseract.image_to_string(imgCrop37)
x_tex_37 = tex_37.rstrip()
tex_333 = pytesseract.image_to_string(imgCrop333)
x_tex_333 = tex_333.rstrip()
tex_444 = pytesseract.image_to_string(imgCrop444)
x_tex_444 = tex_444.rstrip()

with open('DANE_auta2.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file)

    employee_writer.writerow(['ID', 'Data','Oznaczenie dumumentu', 'Nr rejestracyjny', 'Data rejestracji', 'Właściciel pojazdu', 'Adres  wlasciciela','Marka', 'Typ pojazdu', 'Kod CNIT','E - nr identyfikacyjny VIN"','Nazwa Handlowa','F1 - Masa max', 'F-2 - Masa max', 'F-3 - Masa max', 'G - Masa z kierowca', 'G1 - Masa pustego', 'J - Kategoria pojazdu', 'J.1 - Rodzaj pojazdu (fr)', 'J.2 - Typ karoserii (eu)', 'J.3 - Typ karoserii (fr)','K - Nr recepcji (import)', 'P.1 - Pojemność silnika','P.2 - Moc silnika ', 'P.3 - Rodzaj paliwa', 'P.6 - Moc administracyjna','Q - (tylko dla motocykli)','S.1 - Ilość miejsc siedzących','S.2 - Ilość miejsc stojacych','U.1 - Poziom halasu silnika w [dB]','U.2 - predkosc obrotowa silnika','V.7 - Emisja CO2 [g/km]','V.9 - Spełniana forma emisji spalin w/g UE','X.1 - Data ważności przeglądu technicznego','Y.1 - Podatek regionalny:','Y.2 - Podatek dodoatkowy:','Y.3 Podatek ekologiczny:','Y.4 - Podatek za usluge administracyjna','Y.5 -Podatek za wysylke poczta', 'Y.6 - Suma oplat','I - Data wystawienia dowodu', 'Dolna czesc odcieta', 'Kod'])
    employee_writer.writerow(['1', d1,x_tex_0,x_tex_1, x_tex_2, 'x_tex_3', 'x_tex_4',x_tex_5,x_tex_6,x_tex_7,x_tex_E,x_tex_8, x_tex_9, x_tex_10,x_tex_11,x_tex_12,x_tex_13,x_tex_14,x_tex_15,x_tex_16,x_tex_17,x_tex_18,x_tex_19,x_tex_20,x_tex_21,x_tex_22,x_tex_23,x_tex_24,x_tex_25,x_tex_26,x_tex_27,x_tex_28,x_tex_29,x_tex_30,x_tex_31,x_tex_32,x_tex_33,x_tex_34,x_tex_35,x_tex_36,x_tex_37,x_tex_333,x_tex_444])



# --- KONIEC  --- #
cv2.waitKey(0)