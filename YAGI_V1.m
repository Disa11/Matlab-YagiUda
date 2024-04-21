function [] = YAGI_V1
% CODIGO PARA ANTENAS Y RADIO PROGACION STEVEN KNAUTH V1 %
clear all;
clc;
global MMAX NMAX Z RHO N2 NMODE L

M = input ('\n   Numero de modos para cada elemento ->(UN NUMERO POSITIVO): ', 's');
M = str2num(M);
M = round(M);

if (M > 30) % Maximo valor para los modos de una antena Yagi Uda
    fprintf(1, '\n ### NO EXCEDA LOS 30 MODOS ###');
    return
elseif (M <= 0)
    fprintf(1, '\n ### INGRERSE UN M POSITIVO ####\n');
    return
elseif isempty (M) % Si el usuario ingresa cualquier otra cosa que no sea un numero
    fprintf(1, '\n ### INGRESE UN NUMERO POSITIVO PARA M ###\n');
    return   
end

% Numero de elementos de la Antenna 
N = input('   Numero de elementos ->(Numero positivo): ', 's');
N = str2num(N);
N = round(N);

if (N > 20) % Estableciendo un limite de 20 elementos como maximo
    fprintf (1, '\n   ### EL PROGRAMA ADMITE UN MAXMIO DE 20 ELEMENTOS ###\n\n');
    return
elseif (N <= 0)
    fprintf(1, '\n   ###  EL NUMERO DE ELEMENTOS TIENE QUE SER MAYOR A 0 ###\n\n');
    return
elseif isempty(N)   % If the user enters a value other than a number
    fprintf(1, '   ### N TIENE QUE SER UN NUEMRO ###\n\n \n');
    return      
end

% Longitudes de los elementos
if (N > 3)
    ANS = input('   Todos los elementos tiene la misma longitud? (S/N): ', 's');
else 
    ANS = 'N';
end

if (ANS == 's') | (ANS == 'S')
    % Longitud de todos los directores en caso de ser iguales
    LDIR = input('   Ingrese la longitud de todos los directores: ', 's');
    LDIR = str2num(LDIR);
    
    if isempty(LDIR) | (LDIR <= 0)
        fprintf(1, '\n   La longitud no puede ser menos o igual a 0\n\n');
        return;
    end
    % Arreglo de directores en caso de todos tener la misma longitud %
    L = LDIR * ones(1, N-2); % Un arreglo de N-2 elementos

elseif (ANS == 'n') | (ANS == 'N')
    count = 1;
    while count <= (N-2)
        fprintf(1, '   Longitud del director %2d (En longitud de onda):', count);
        L_DIR_L = input('', 's');
        L_DIR_L = str2num(L_DIR_L);
        
        if (isempty(L_DIR_L)) | (L_DIR_L <= 0)
            fprintf(1, '\n   INGRESE UNA LONGITUD CORRECTA');
            return;
        end
        L (count) = L_DIR_L;
        count = count + 1;
    end
else
    fprintf(1, '\n   La respuesta no tiene el formato adecuado (S/N) o (s/n)');
    return;
end

% Longitud del Reflector N - 1 %
L_REF = input('   Longitud del Reflector (En Longitud de onda)', 's');
L_REF = str2num(L_REF);
if (isempty(L_REF)) | (L_REF <= 0)
    fprintf(1, '\n    ### INGRESE UNA LONGITUD CORRECTA ###');
    return;
end
L (N-1) = L_REF;

% Longitud del Dipolo N %
L_DIPOLO = input('\n   Longitud del Dipolo (En Longitud de onda)', 's');
L_DIPOLO = str2num(L_DIPOLO);
if (isempty(L_DIPOLO)) | (L_DIPOLO <= 0)
    fprintf(1, '    ### INGRESE UNA LONGITUD CORRECTA ###');
    return;
end
L (N) = L_DIPOLO;

% Serparaciones entre elementos %
SEP_DIP_DIR1 = input('\n    Separacion entre el Dipolo y el primer director','s');
SEP_DIP_DIR1 = str2num(SEP_DIP_DIR1);
if (isempty(SEP_DIP_DIR1)) | (SEP_DIP_DIR1 <= 0)
    fprintf(1, '    ### INGRESE UNA SEPARACION CORRECTA ###');
    return;
end
S_1 = SEP_DIP_DIR1;
fprintf(1, '\n');

if (N > 3)
    ANS = input('   La separacion entre elementos es uniforme? (S/N): ', 's');
else 
    ANS = 'N';
end

if (ANS == 's') | (ANS == 'S')
    % Separacion de todos los directores en caso de ser uniforme
    SDIR = input('   Ingrese la separacion entre directores (en longitud de ondas): ', 's');
    SDIR = str2num(SDIR);
    
    if isempty(SDIR) | (SDIR <= 0)
        fprintf(1, '\n   La separacion no puede ser menos o igual a 0\n\n');
        return;
    end
    % Arreglo de la separaciondirectores en caso de ser uniforme %
    S = SDIR * ones(1, N-2); % Un arreglo de N-2 elementos

elseif (ANS == 'n') | (ANS == 'N')
    count = 2;
    while count <= (N-2)
        fprintf(1, '   Separacion entre el director # %2d y el # %2d (En longitud de onda):', count - 1, count);
        S_DIR_L = input('', 's');
        S_DIR_L = str2num(S_DIR_L);
        
        if (isempty(S_DIR_L)) | (S_DIR_L <= 0)
            fprintf(1, '\n   INGRESE UNA SEPARACION CORRECTA');
            return;
        end
        S (count) = S_DIR_L;
        count = count + 1;
    end
else
    fprintf(1, '\n   La respuesta no tiene el formato adecuado (S/N) o (s/n)');
    return;
end
S(1) = S_1;

% separacion entre el reflector y el dipolo %
SEP_DIP_REF = input('\n   Separacion entre el Reflector y el dipolo: ','s');
SEP_DIP_REF = str2num(SEP_DIP_REF);
if (isempty(SEP_DIP_REF)) | (SEP_DIP_REF <= 0)
    fprintf(1, '    ### INGRESE UNA SEPARACION CORRECTA ###');
    return;
end
S (N-1) = SEP_DIP_REF;

% Radio de los elementos %
R_ELM = input('\n   Radio de los elementos (En longitud de ondas): ','s');
R_ELM = str2num(R_ELM);
if (isempty(R_ELM)) | (R_ELM <= 0)
    fprintf(1, '    ### INGRESE UN RADIO CORRECTO ###');
    return;
end
ALPHA = R_ELM;

% Frecuencias subportadoras %
f1 = input('\n   Frecuencia Subportadora 1: ','s');
f1 = str2num(f1);
if (isempty(f1)) | (f1 <= 0)
    fprintf(1, '    ### INGRESE UNA FRECUENCIA CORRECTA ###');
    return;
end

f2 = input('\n   Frecuencia Subportadora 2: ','s');
f2 = str2num(f2);
if (isempty(f2)) | (f2 <= 0)
    fprintf(1, '    ### INGRESE UNA FRECUENCIA CORRECTA ###');
    return;
end

% ------------ El calculo real inicia aca ------------ %
% Inicializar variables %
f = (f1 + f2) / 2;
% DUDAS SERIAS %
lambda = 299792458/f;
%lambda = 1;
a = 1;
while a <= (N - 2)
    YP (a) = a * S (a);
    a = a + 1;
end
YP (N-1) = - S (N-1);
YP (N) = 0;

RES = 0;
G2 = 0;
INDEX = 0;
DZ = L / (2 * M - 1);
ETA = 120 * pi;
MU = 4 * pi * 10 ^ (-7);
C = 3 * 10 ^ 8;
K = (2 * pi)/ lambda;
RTOD = 180 / pi;
DTOR = pi / 180;

A = zeros (M * N, M * N);
B = 1:(M*N);
B = B * 0;
Inm = zeros (N, M);

I = 1; 

% Calculos reales %
h=waitbar(0,'El programa esta en ejecucion ...');
while I <= (M * N)
    waitbar(0.8*I/(M*N),h);
    IFACT = floor ((I - 1) / M);%     Esto determina la posición del observador con
    N1 = IFACT + 1;             %  N1 siendo el elemento en el que se encuentra el observador.
    IMODE = I - IFACT * M;
    
    Z = (M - IMODE) * DZ (N1);  %  Basado en el número de modo y elemento, encuentra la distancia Z
    
    J = 1;
    while J <= (M * N)           
    
        JFACT = floor ((J - 1) / M); %     Esto determina la posición de la fuente y su
        N2 = JFACT + 1;         %  modo correspondiente. N2 es el elemento en el que se
        NMODE = J - JFACT * M;  %  encuentra la fuente y NMODE es su número de modo.
        
        if (N1 == N2)           %     Si se encuentra el efecto de un modo en el elemento
            RHO = ALPHA;        %  en el que está ubicado, Y es el radio del elemento.
        else                    %  De lo contrario, la distancia Y se encuentra usando la fórmula
            RHO = YP (N1) - YP (N2);    %  (asumiendo X, X' = 0).
        end
        
        LL = 0;                 %     Define los límites de la integración
        UL = L (N2) / 2;

        RES = SINTEG (UL, LL, 10, lambda);  %     Realiza la integración numérica
        LEN = L (N2) / 2;
        G2 = KERNEL (LEN, lambda);
        F2M = NMODE * 2 - 1;

        A (I, J) = ETA / (j * 8 * pi ^ 2) * ...
                     ((F2M * pi / L (N2)) * (-1) ^ (NMODE + 1) * G2 + ...
                      (K ^ 2 - F2M ^ 2 * pi ^ 2 / L (N2) ^2) * RES);
        J = J + 1;

    end
    
    I = I + 1;
end

% Llena la última fila de la matriz correspondiente al alimentador.
I = zeros(1, M * (N - 1));
J = ones(1, M);
A(M * N, :) = [I J];
B(M * N) = 1;

% Invierte el sistema para resolver los coeficientes de corriente en la
% expansión de la serie de Fourier.
ISIZE = N * M;
[A, IPERM, PIVOT] = LUDEC(A, ISIZE);
B = LUSOLV(A, ISIZE, IPERM, B);

% Convierte el array único de coeficientes de corriente a un array doble de
% la forma Imn.
NCUT = 0;
I = 1;
while I <= N
    J = 1;
    while J <= M
        Inm(I, J) = B(J + NCUT);
        J = J + 1;
    end
    NCUT = NCUT + M;
    I = I + 1;
end

% Calcula los campos radiados en el plano E.
% En este plano, Theta varía de 0 a 180, mientras que Phi es 90 en la
% mitad del patrón y 270 en la otra mitad. El patrón se calcula en
% incrementos de un grado.
NCUT = 0;
ML = 1;
while ML <= 2
    if (ML == 1)
        PHI = 90 * DTOR;
        MAX = 181;
    else
        PHI = 270 * DTOR;
        MAX = 180;
    end
    ICOUNT = 1;
    while ICOUNT <= MAX
        waitbar(ICOUNT/MAX*ML*0.5*0.2+0.8,h);
        THETA = (ICOUNT - 1) * DTOR;
        if (THETA > pi)
            PHI = 270 * DTOR;
        end
        EZP = 0;
        I = 1;
        while I <= N
            IZP = 0;
            J = 1;
            while J <= M
                MODE = J;
                LEN = L(I);
                ANG = THETA;
                IZP = IZP + Inm(I, J) * ...
                            (ZMINUS(ANG, LEN, MODE, lambda) + ...
                             ZPLUS(ANG, LEN, MODE, lambda));
                J = J + 1;
            end
            AEXP = K * YP(I) * sin(THETA) * sin(PHI);
            EZP = EZP + L(I) * exp(j * AEXP) * IZP;
            I = I + 1;
        end
        ETHETA(NCUT + ICOUNT) = j * C * MU / 8 * sin(THETA) * EZP;
        ICOUNT = ICOUNT + 1;
    end
    NCUT = NCUT + MAX;
    ML = ML + 1;
end
close(h);

% Encuentra el valor máximo en el patrón en el plano E.
EMAX = 10 ^ (-12);
abs_ETHETA = abs(ETHETA);
ARG = max(abs_ETHETA);
if ARG > EMAX
    EMAX = ARG;
end

% Normaliza el patrón al valor máximo, convierte a dB.
I = 1;
while I <= 361
    THETA = I - 1;
    ARG = abs(ETHETA(I));
    if ((ARG/EMAX) > (10 ^ (-6)))
        ETH(I) = 20 * log10(ARG / EMAX);
    else
        ETH(I) = -120;
    end
    I = I + 1;
end
% Registra los valores del plano E para futuras representaciones gráficas.
E_PLANE = ETH;

% Encuentra la relación frontal-a-tras en el patrón en el plano E.
EFTOB = -ETH(271);

% Encuentra el ancho de haz de 3 dB en el patrón en el plano E.
I = 91;
while I <= 270
    ETH(I) = ETH(I) + 3;
    I = I + 1;
end
I = 91;
while I <= 270
    THETA = I-1;
    if (ETH(I) == 0)
        E3D_BW = 2 * ((I - 1) - 90);
        I = 300;    % Termina el bucle while temprano
    elseif ((ETH(I - 1) > 0) & (ETH(I) < 0))
        E3D_BW = 2 * (- ETH(I) / (ETH(I) - ETH(I - 1)) + (I - 1) - 90);
        I = 300;
    end
    I = I + 1;
end

% Calcula los campos radiados en el plano H.
% En este plano, Theta = 90 y Phi varía de 0 a 360.
THETA = 90 * DTOR;
MAX = 361;
ICOUNT = 1;
while ICOUNT <= MAX
    PHI = (ICOUNT - 1) * DTOR;
    EZP = 0;
    I = 1;
    while I <= N
        IZP = 0;
        J = 1;
        while J <= M
            MODE = J;
            LEN = L(I);
            ANG = PHI;
            IZP = IZP + Inm(I, J) * ...
                        (ZMINUS(ANG, LEN, MODE, lambda) + ...
                         ZPLUS(ANG, LEN, MODE, lambda));
            J = J + 1;
        end
        AEXP = K * YP(I) * sin(THETA) * sin(PHI);
        EZP = EZP + L(I) * exp(j * AEXP) * IZP;
        I = I + 1;
    end
    ETHETA(ICOUNT) = j * C * MU / 8 * sin(THETA) * EZP;
    ICOUNT = ICOUNT + 1;
end

% Encuentra el valor máximo en el patrón en el plano H.
EMAX = 10 ^ (-12);
abs_ETHETA = abs(ETHETA);
ARG = max(abs_ETHETA);
if (ARG > EMAX)
    EMAX = ARG;
end


I = 1;
while I <= 361
    PHI = I - 1;
    ARG = abs(ETHETA(I));
    if (ARG / EMAX) > (10 ^ (-6))
        ETH(I) = 20 * log10(ARG / EMAX);
    else
        ETH(I) = -120;
    end
    I = I + 1;
end

% REGISTRAR LOS VALORES EN EL PLANO H PARA FUTURAS GRÁFICAS
H_PLANE = ETH;

% CALCULAR LA RELACIÓN FRONTAL-TRASERA EN EL PATRÓN DEL PLANO H
HFTOB = -ETH(271);

% CALCULAR EL ANCHO DEL HAZ A -3 dB EN EL PATRÓN DEL PLANO H
I = 1;
while I <= 181
    ETH(I) = ETH(I) + 3;
    I = I + 1;
end

I = 91;
while I <= 270
    PHI = I - 1;
    if (ETH(I) == 0)
        H3D_BW = 2 * ((I - 1) - 90);
        I = 300;
    elseif (ETH(I - 1) > 0) & (ETH(I) < 0)
        H3D_BW = 2 * (-ETH(I) / (ETH(I) - ETH(I - 1)) + (I - 1) - 90);
        I = 300;
    end
    I = I + 1;
end

% CALCULAR LA DIRECTIVIDAD DE LA ANTENA
THETA = 90 * DTOR;
PHI = 90 * DTOR;
AZ = 0;

I = 1;
while I <= N
    IZP = 0;
    J = 1;
    while J <= M
        MODE = J;
        LEN = L(I);
        ANG = THETA;
        IZP = IZP + Inm(I, J) * ...
                    (ZMINUS(ANG, LEN, MODE, lambda) + ...
                     ZPLUS(ANG, LEN, MODE, lambda));
        J = J + 1;
    end
    AEXP = K * YP(I) * sin(THETA) * sin(PHI);
    AZ = AZ + L(I) * exp(j * AEXP) * IZP;
    I = I + 1;
end

UMAX = 3.75 * pi * abs(AZ)^2 * sin(THETA)^2;

PRAD = SCINT2(0, pi, 0, 2 * pi, N, M, Inm, YP, lambda);

D0 = 4 * pi * UMAX / abs(PRAD);

fprintf(1, '\n   ----------------------------------------------------\n');
fprintf(1, '   Resultados del programa Antena Yagi Uda \n');
fprintf(1, '   ------------------------------------------------------\n');
fprintf(1, '\n   ANCHO DE BANDA DE 3 dB EN EL PLANO E = %12.2f  GRADOS\n', E3D_BW);
fprintf(1, '\n   ANCHO DE BANDA DE 3 dB EN EL PLANO H = %12.2f  GRADOS\n', H3D_BW);
fprintf(1, '\n   RELACIÓN FRONTAL-A-TRASERA EN EL PLANO E = %12.4f  dB\n', EFTOB);
fprintf(1, '\n   RELACIÓN FRONTAL-A-TRASERA EN EL PLANO H = %12.4f  dB\n', HFTOB);
fprintf(1, '\n   DIRECTIVIDAD = %16.3f  dB\n', 10 * log10(D0));

%Basado en los coeficientes de Fourier de la corriente, 
%calcula la distribución de corriente en los elementos. 
%Ten en cuenta que cada elemento se subdivide en 100 secciones para este cálculo.%

IL = 1;
while IL <= N   
    DZ(IL) = L(IL) / 100; % Calcula el tamaño de cada sección del elemento
    
    I = 1;
    while I <= 51
        
        Z = (I - 1) * DZ(IL); % Calcula la posición a lo largo del elemento
        IZP = 0;
        
        J = 1;
        while J <= M
            
            F2M = 2 * J - 1;
            IZP = IZP + Inm(IL, J) * cos(F2M * pi * Z / L(IL)); % Calcula la corriente en la posición Z
            
            J = J + 1;
        end
        
        CUR(I) = abs(IZP); % Magnitud de la corriente en la posición Z
        angle = atan2(imag(IZP), real(IZP)); 
        PHA(I) = angle * RTOD; % Fase de la corriente en la posición Z
        
        I = I + 1;
    end

    I = 1;
    while I <= 51
        Z = (I - 1) * DZ(IL);
        I = I + 1;
    end

    CENTER_CURRENT(IL) = CUR(1); % Registra para futuras representaciones gráficas
    IL = IL + 1;
end

I = 1;
while I <= N
    J = 1;
    while J <= M
        
        CURRENT = abs(Inm(I, J)); % Magnitud de la corriente para el modo actual
        angle_radian = atan2(imag(Inm(I, J)), real(Inm(I, J)));
        ANGLE = angle_radian * RTOD; % Fase de la corriente para el modo actual
        
        J = J + 1;
    end
    
    I = I + 1;
end

E_PLANE = E_PLANE (1:360);
H_PLANE = H_PLANE (1:360);
angle = 1:1:360;

% Graficar el patrón de campo en los planos E y H
figure;
plot(angle, E_PLANE, '-b', 'LineWidth', 2); % Grafica el patrón en el plano E
hold on;
plot(angle, H_PLANE, '--r', 'LineWidth', 2); % Grafica el patrón en el plano H
legend('Plano E', 'Plano H'); % Agrega leyenda
xlim([1 360]); % Establece los límites del eje x
ylim([-60 0]); % Establece los límites del eje y
title('Análisis de Yagi-Uda'); % Establece el título del gráfico
xlabel('Theta(E)/Phi(H) grados'); % Etiqueta del eje x
ylabel('Patrón de campo (dB)'); % Etiqueta del eje y
hold off;

% Graficar la distribución de corriente
figure;
INDEX = 1:1:N;
if N >= 3
    CENTER_CURRENT = [CENTER_CURRENT(N-1:N) CENTER_CURRENT(1:N-2)];
    INDEX = [INDEX(N-1:N) INDEX(1:N-2)]; 
end
plot(CENTER_CURRENT, 'LineWidth', 2); % Grafica la distribución de corriente
set(gca, 'XTick', 1:1:N); % Establece las marcas en el eje x
set(gca, 'XTickLabel',INDEX); % Etiqueta las marcas en el eje x
title('Distribución de corriente'); % Establece el título del gráfico
ylim([0 1]); % Establece los límites del eje y
xlabel('Número de elemento'); % Etiqueta del eje x
ylabel('Amplitud de corriente del elemento'); % Etiqueta del eje y

% Graficar los patrones de elevación en los planos E y H
figure;
h1 = elevation(angle*pi/180, E_PLANE, -40, 0, 5, 'b'); hold on; % Grafica el patrón de elevación en el plano E
h2 = elevation(angle*pi/180, H_PLANE, -40, 0, 5, 'r--'); % Grafica el patrón de elevación en el plano H
set([h1 h2], 'linewidth', 2); % Establece el ancho de línea
legend([h1 h2], {'Plano E', 'Plano H'}); % Agrega leyenda

end

%     FUNCIÓN SINTEG (PRECISIÓN SIMPLE)
%
%     PROPÓSITO
%     REALIZAR INTEGRACIÓN COMPLEJA SIMPLE
%     REALIZA INTEGRACIÓN POR CUADRATURA GAUSSIANA DE DIECISÉIS PUNTOS
%     CON PRECISIÓN CRECIENTE ESTABLECIDA POR ENTERO NO
%
%     USO
%     ANS = SINTEG(UL, LL, NO)
%
%     DESCRIPCIÓN DE LOS PARÁMETROS
%     UL  -  LÍMITE SUPERIOR DE LA INTEGRACIÓN (REAL)
%     LL  -  LÍMITE INFERIOR DE LA INTEGRACIÓN (REAL)
%     NO  -  NÚMERO DE DIVISIONES ENTRE LL Y UL  (ENTERO)
%
%     ANS -  RESULTADO DE LA INTEGRACIÓN
%     L   -  LAS LONGITUDES DE LOS ELEMENTOS DEL ARREGLO (VARIABLE GLOBAL)
%
function ANS = SINTEG(UL, LL, NO, lambda)

% Coeficientes de cuadratura gaussiana y pesos asociados
GAUSS = [-0.0950125098376370
         -0.2816035507792590
         -0.4580167776572270
         -0.6178762444026440
         -0.7554044083550030
         -0.8656312023878320
         -0.9445750230732330
         -0.9894009349916500
          0.0950125098376370
          0.2816035507792590
          0.4580167776572270
          0.6178762444026440
          0.7554044083550030
          0.8656312023878320
          0.9445750230732330
          0.9894009349916500];
     
LEGEND = [0.1894506104550680
          0.1826034150449240
          0.1691565193950020
          0.1495959888165770
          0.1246289712555340
          0.0951585116824930
          0.0622535239386480
          0.0271524594117540
          0.1894506104550680
          0.1826034150449240
          0.1691565193950020
          0.1495959888165770
          0.1246289712555340
          0.0951585116824930
          0.0622535239386480
          0.0271524594117540];

% Calcula el tamaño de paso
DEL = (UL - LL) / (2 * NO);
SUM = 0;

% Realiza la integración usando cuadratura gaussiana
J = 1;
while J <= NO
    S = LL + (2 * J - 1) * DEL;
    I = 1;
    while I <= 16
        X = S + GAUSS(I) * DEL;
        SUM = SUM + LEGEND(I) * FF(X, lambda); % Evalúa la función FF en el punto X y suma ponderada
        I = I + 1;
    end
    J = J + 1;
end

% Multiplica por el tamaño de paso para obtener el resultado final de la integración
ANS = SUM * DEL;

% Fin de la función SINTEG
end

%     FUNCIÓN FF
function value = FF(X, lambda)
global Z RHO N2 NMODE L

K = (2 * pi)/lambda;

RMINUS = sqrt(RHO ^ 2 + (Z - X) ^ 2);
RPLUS = sqrt(RHO ^ 2 + (Z + X) ^ 2);

F2M = NMODE * 2 - 1;

% Calcula el valor de la función FF en el punto X
value = (exp(-j * K * RMINUS) / (4 * pi * RMINUS) + ...
         exp(-j * K * RPLUS) / (4 * pi * RPLUS)) * ... 
        cos(F2M * pi * X / L(N2));

%  Fin de la función FF
end

%     FUNCIÓN KERNEL
function G2 = KERNEL(ZP, lambda)
global Z RHO
     
K = (2 * pi)/lambda;

RMINUS = sqrt(RHO ^ 2 + (Z - ZP) ^ 2);
RPLUS = sqrt(RHO ^ 2 + (Z + ZP) ^ 2);

G2 = exp(-j * K * RMINUS) / (4 * pi * RMINUS) + ...
     exp(-j * K * RPLUS) / (4 * pi * RPLUS);

%  Fin de la función KERNEL
end

%     FUNCIÓN ZMINUS
function value = ZMINUS(TH, LG, NM, lambda)

K = (2 * pi)/lambda;

F2M = 2 * NM - 1;

ARG1 = (F2M * pi / LG - K * cos(TH)) * (LG / 2);
if (ARG1 == 0)
    value = 1;
else
    value = sin(ARG1) / ARG1;
end

%  Fin de la función ZMINUS
end

%     FUNCIÓN ZPLUS
function value = ZPLUS(TH, LG, NM, lambda)

K = (2 * pi)/lambda;

F2M = 2 * NM - 1;

ARG2 = (F2M * pi / LG + K * cos(TH)) * (LG / 2);
if (ARG2 == 0)
    value = 1;
else
    value = sin(ARG2) / ARG2;
end

%  Fin de la función ZPLUS
end

%****************************************************************
%     FUNCIÓN FXZ
%
function value = FXZ(THETA, PHI, N, M, Inm, YP, lambda)
global L

%  ---------------- Cambiar Lambda --------- %
K = (2 * pi)/0.1397633837;
AZ = 0;

I = 1;
while I <= N
    
    IZP = 0;
    
    J = 1;
    while J <= M
        
        MODE = J;
        LEN = L(I);
        ANG = THETA;
        
        IZP = IZP + Inm(I, J) * ...
                    (ZMINUS(ANG, LEN, MODE, lambda) + ...
                     ZPLUS(ANG, LEN, MODE, lambda));
        J = J + 1;         
    end
    
    AEXP = K * YP(I) * sin(THETA) * sin(PHI);
    AZ = AZ + L(I) * exp(j * AEXP) * IZP;
    
    I = I + 1;
end

U = 3.75 * pi * abs(AZ) ^ 2 * sin(THETA) ^ 2;

% Calcula el valor de la función FXZ
value = sin(THETA) * U;

%  Fin de la función FXZ
%****************************************************************

end

%****************************************************************
%     FUNCIÓN LUDEC
%
function [A, IPERM, SCAL] = LUDEC(Z, N)

%    REEMPLAZA LA MATRIZ POR SU DECOMPOSICIÓN LU
%    OBTIENE INFORMACIÓN DE ESCALADO.

I = 1;
while I <= N
    
    ZMAX = 0;
    
    J = 1;
    while J <= N
        
        CAZ = abs(Z(I, J));
        if (CAZ > ZMAX) 
            ZMAX = CAZ;
        end
        J = J + 1;
    end
    
    SCAL(I) = 1 / ZMAX;
    I = I + 1;
end

%    Algoritmo de Crout.
J = 1;
while J <= N

    I = 1;
    while I <= (J - 1)
        
        K = 1;
        while K <= (I - 1)
            Z(I, J) = Z(I, J) - Z(I, K) * Z(K, J);
            K = K + 1;
        end
        
        I = I + 1;
    end

%    BUSCA EL ELEMENTO PIVOTE MÁS GRANDE.
    ZMAX = 0;
    
    I = J;
    while I <= N
        
       K = 1;
       while K <= (J - 1)
           Z(I, J) = Z(I, J) - Z(I, K) * Z(K, J);
           K = K + 1;
       end
        
       TEST = SCAL(I) * abs(Z(I, J));
       
       if (round((TEST - ZMAX) * 10 ^ 8) > 0)  % Método temporal
           IMAX = I;
           ZMAX = TEST;
       end

       I = I + 1;
    end

%    INTERCAMBIA LAS FILAS.
    if (J ~= IMAX)
         
        K = 1;
        while K <= N
            TEMP = Z(IMAX, K);
            Z(IMAX, K) = Z(J, K);
            Z(J, K) = TEMP;
            K = K + 1;
        end         
        SCAL(IMAX) = SCAL(J);
    end
   
%   DIVIDE POR EL ELEMENTO PIVOTE.
    IPERM(J) = IMAX;
    if (J ~= N)
         
        I = J + 1;
        while I <= N
            Z(I, J) = Z(I, J) / Z(J, J);
            I = I + 1;
        end
    end
     
    J = J + 1;
end

A = Z;

%  Fin de la función LUDEC
%****************************************************************


end
%****************************************************************
%     FUNCIÓN LUSOLV
%
function B = LUSOLV(Z, N, IPERM, V)

%    RESUELVE EL SISTEMA LINEAL DADA LA DECOMPOSICIÓN LU DE LUDEC
%    EL VECTOR DE FORZAMIENTO SE REEMPLAZA CON EL VECTOR SOLUCIÓN AL SALIR

%    SUSTITUCIÓN HACIA ADELANTE.
I = 1;
while I <= N
    
    TEMP = V(IPERM(I));
    V(IPERM(I)) = V(I);
    
    J = 1;
    while J <= (I - 1)
        TEMP = TEMP - Z(I, J) * V(J);
        J = J + 1;
    end
    
    V(I) = TEMP;
    I = I + 1;
end

%    SUSTITUCIÓN HACIA ATRÁS.
I = 1;
while I <= N
    II = N - I + 1;
    TEMP = V(II);
    
    J = II + 1;
    while J <= N
        TEMP = TEMP - Z(II, J) * V(J);
        J = J + 1;
    end
    
    V(II) = TEMP / Z(II, II);
    I = I + 1;
end

B = V;
%  Fin de la función LUSOLV
%****************************************************************
end

%******************************************************************
% FUNCIÓN SCINT2
% SCINT2 ES UNA RUTINA DE INTEGRACIÓN DE PRECISIÓN SIMPLE Y COMPLEJA
% EN 2 DIMENSIONES. ESTA RUTINA UTILIZA CUADRATURAS GAUSSIANAS DE 16 PUNTOS,
% CON COEFICIENTES DE LEGENDRE. ENTRA CON:
%
%       (R)X1          LÍMITE INFERIOR DE LA INTEGRACIÓN EN X
%       (R)X2          LÍMITE SUPERIOR DE LA INTEGRACIÓN EN X
%       (R)Z1          LÍMITE INFERIOR DE LA INTEGRACIÓN EN Z
%       (R)Z2          LÍMITE SUPERIOR DE LA INTEGRACIÓN EN Z
%       (C)RES         RESULTADOS DE LA INTEGRACIÓN
%       (I)N           NÚMERO DE ELEMENTOS EN EL ARRAY
%       (I)M           NÚMERO DE MODOS POR ELEMENTO
%       (R)Inm         EL ARRAY DE COEFICIENTES DE CORRIENTE
%       (R)YP          LAS DISTANCIAS DE LOS ELEMENTOS A LO LARGO DEL EJE Y
%
%***********************************************************************
%
%
function RES = SCINT2(X1, X2, Z1, Z2, N, M, Inm, YP, lambda)

R = [0.0950125098
     0.2816035508
     0.4580167777
     0.6178762444
     0.7554044084
     0.8656312024
     0.9445750231
     0.9894009350
     -0.9894009350
     -0.9445750231
     -0.8656312024
     -0.7554044084
     -0.6178762444
     -0.4580167777
     -0.2816035508
     -0.0950125098];
 
W = [0.1894506105
     0.1826034150
     0.1691565194
     0.1495959888
     0.1246289713
     0.0951585117
     0.0622535239
     0.0271524594
     0.0271524594
     0.0622535239
     0.0951585117
     0.1246289713
     0.1495959888
     0.1691565194
     0.1826034150
     0.1894506105];
 
SX = 0.5 * (X2 + X1);
DX = 0.5 * (X2 - X1);
SZ = 0.5 * (Z2 + Z1);
DZ = 0.5 * (Z2 - Z1);

TT = 0;

J = 1;
while J <= 16
    
    Z  = SZ + DZ * R(J);
    SS = 0;
    
    I = 1;
    while I <= 16
        
        X  = SX + DX * R(I);
        SS = W(I) * FXZ(X, Z, N, M, Inm, YP, lambda) + SS;
        
        I = I + 1;
    end
    
    S = DX * SS;
    TT = S * W(J) + TT;
    
    J = J + 1;
end

% Calcula el resultado de la integración SCINT2
RES = DZ * TT;
%  Fin de la función SCINT2
%****************************************************************

end

%***********************************************************************
%       elevation(theta, rho, rmin, rmax, rticks, line_style)
%**********************************************************************
%       GAINPLOT realiza un gráfico de ganancia de antena utilizando coordenadas polares del ángulo THETA, en radianes, 
%       versus el radio RHO, donde RHO puede ser negativo.
%
%     - rmin establece el valor del centro del gráfico. (por ejemplo, -40 dB)
%     - rmax establece el valor del anillo exterior del gráfico (por ejemplo, 0 dB)
%     - rticks es el número de marcas radiales (o círculos) que desea.
%              NO ES EL ESPACIAMIENTO ENTRE ELLOS. Además, si rticks es un número par > 5, entonces se dividirá por 2, 
%              o si rticks > 5 y es divisible por 3, rticks se dividirá por 3.
%     - line_style es sólido o discontinuo, etc. (el predeterminado es una línea amarilla sólida)
%
%       POLAR(THETA, RHO, S) utiliza el estilo de línea especificado en la cadena S.
%       Consulte PLOT para obtener una descripción de los estilos de línea legales.
%       Consulte también PLOT, LOGLOG, SEMILOGX, SEMILOGY.
%********************************************************************
%       Créditos:
%		S. Bellofiore
%		S. Georgakopoulos
%		A. C. Polycarpou
%		C. Wangsvick
%		C. Bishop
%
%	Tabule sus datos en consecuencia y llame a polar_dB para proporcionar el
%	Gráfico polar 2D
%
%	Nota: Esta función es diferente de polar.m (proporcionada por MATLAB) porque RHO se da en dB y puede ser negativa
%-----------------------------------------------------------------------------

function hpol = elevation(theta, rho, rmin, rmax, rticks, line_style)

if nargin < 5
    error('Requiere 5 o 6 argumentos de entrada.')
elseif nargin == 5
    if isstr(rho)
        line_style = rho;
        rho = theta;
        [mr, nr] = size(rho);
        if mr == 1
            theta = 1:nr;
        else
            th = (1:mr)';
            theta = th(:, ones(1, nr));
        end
    else
        line_style = 'auto';
    end
elseif nargin == 1
    line_style = 'auto';
    rho = theta;
    [mr, nr] = size(rho);
    if mr == 1
        theta = 1:nr;
    else
        th = (1:mr)';
        theta = th(:, ones(1, nr));
    end
end
if isstr(theta) || isstr(rho)
    error('Los argumentos de entrada deben ser numéricos.');
end
if any(size(theta) ~= size(rho))
    error('THETA y RHO deben ser del mismo tamaño.');
end

% obtener el estado de retención
cax = newplot;
next = lower(get(cax, 'NextPlot'));
hold_state = ishold;

% obtener el color del texto del eje x para que la cuadrícula esté en el mismo color
tc = get(cax, 'xcolor');

% Mantener los valores predeterminados de Texto actuales, restablecerlos a los atributos de fuente del Ejes para que las marcas de graduación los utilicen.
fAngle = get(cax, 'DefaultTextFontAngle');
fName = get(cax, 'DefaultTextFontName');
fSize = get(cax, 'DefaultTextFontSize');
fWeight = get(cax, 'DefaultTextFontWeight');
set(cax, 'DefaultTextFontAngle', get(cax, 'FontAngle'), ...
    'DefaultTextFontName', get(cax, 'FontName'), ...
    'DefaultTextFontSize', get(cax, 'FontSize'), ...
    'DefaultTextFontWeight', get(cax, 'FontWeight'))

% solo hacer cuadrículas si hold está desactivado
if ~hold_state

    % hacer una cuadrícula radial
    hold on;
    % v devuelve los límites del eje
    % cambié la siguiente línea para permitir que los límites y estén en negativo
    hhh = plot([0 max(theta(:))], [min(rho(:)) max(rho(:))]);
    v = [get(cax, 'xlim') get(cax, 'ylim')];
    ticks = length(get(cax, 'ytick'));
    delete(hhh);

    % verificar límites radiales (rticks)
    if rticks > 5 % ver si podemos reducir el número
        if rem(rticks, 2) == 0
            rticks = rticks / 2;
        elseif rem(rticks, 3) == 0
            rticks = rticks / 3;
        end
    end

    % definir un círculo
    th = 0:pi / 50:2 * pi;
    xunit = cos(th);
    yunit = sin(th);
    % ahora forzar realmente que los puntos en los ejes x / y estén en ellos exactamente
    inds = [1:(length(th) - 1) / 4:length(th)];
    xunits(inds(2:2:4)) = zeros(2, 1);
    yunits(inds(1:2:5)) = zeros(3, 1);

    rinc = (rmax - rmin) / rticks;

    % etiquetar r
    % cambiar la siguiente línea para que el círculo unitario no se multiplique
    % por un número negativo. Lo mismo para las ubicaciones de texto.
    for i = (rmin + rinc):rinc:rmax
        is = i - rmin;
        plot(xunit * is, yunit * is, '-', 'color', tc, 'linewidth', 0.5);
        text(0, is + rinc / 20, ['  ' num2str(i)], 'verticalalignment', 'bottom');
    end
    % trazar radios
    th = (1:6) * 2 * pi / 12;
    cst = cos(th);
    snt = sin(th);
    cs = [-cst; cst];
    sn = [-snt; snt];
    plot((rmax - rmin) * cs, (rmax - rmin) * sn, '-', 'color', tc, 'linewidth', 0.5);

    % trazar las marcas
    george = (rmax - rmin) / 30; % Longitud de las marcas
    th2 = (0:36) * 2 * pi / 72;
    cst2 = cos(th2);
    snt2 = sin(th2);
    cs2 = [(rmax - rmin - george) * cst2; (rmax - rmin) * cst2];
    sn2 = [(rmax - rmin - george) * snt2; (rmax - rmin) * snt2];
    plot(cs2, sn2, '-', 'color', tc, 'linewidth', 0.15); % 0.5
    plot(-cs2, -sn2, '-', 'color', tc, 'linewidth', 0.15); % 0.5

    % anotar radios en grados
    % Cambié la siguiente línea para hacer que los radios sean lo suficientemente largos
    rt = 1.1 * (rmax - rmin);
    for i = 1:max(size(th))
        text(rt * cst(i), rt * snt(i), int2str(abs(i * 30 - 90)), 'horizontalalignment', 'center');
        if i == max(size(th))
            loc = int2str(90);
        elseif i * 30 + 90 <= 180
            loc = int2str(i * 30 + 90);
        else
            loc = int2str(180 - (i * 30 + 90 - 180));
        end
        text(-rt * cst(i), -rt * snt(i), loc, 'horizontalalignment', 'center');
    end
    % establecer vista en 2D
    view(0, 90);

    % establecer límites del eje
    % Cambié la siguiente línea para escalar las cosas correctamente
    axis((rmax - rmin) * [-1 1 -1.1 1.1]);
end

% Restaurar predeterminados.
set(cax, 'DefaultTextFontAngle', fAngle, ...
    'DefaultTextFontName', fName, ...
    'DefaultTextFontSize', fSize, ...
    'DefaultTextFontWeight', fWeight);

% transformar datos a coordenadas cartesianas.
% cambié la siguiente línea para que los rho negativos no se tracen en el otro lado
for i = 1:length(rho)
    if (rho(i) > rmin)
        if theta(i) * 180 / pi >= 0 && theta(i) * 180 / pi <= 90
            xx(i) = (rho(i) - rmin) * cos(pi / 2 - theta(i));
            yy(i) = (rho(i) - rmin) * sin(pi / 2 - theta(i));
        elseif theta(i) * 180 / pi >= 90
            xx(i) = (rho(i) - rmin) * cos(-theta(i) + pi / 2);
            yy(i) = (rho(i) - rmin) * sin(-theta(i) + pi / 2);
        elseif theta(i) * 180 / pi < 0
            xx(i) = (rho(i) - rmin) * cos(abs(theta(i)) + pi / 2);
            yy(i) = (rho(i) - rmin) * sin(abs(theta(i)) + pi / 2);
        end
    else
        xx(i) = 0;
        yy(i) = 0;
    end
end

% trazar datos encima de la cuadrícula
if strcmp(line_style, 'auto')
    q = plot(xx, yy);
else
    q = plot(xx, yy, line_style);
end
if nargout > 0
    hpol = q;
end
if ~hold_state
    axis('equal'); axis('off');
end

% restablecer estado de retención
if ~hold_state, set(cax, 'NextPlot', next); end
end