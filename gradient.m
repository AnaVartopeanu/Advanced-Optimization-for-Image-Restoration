clc;
clear;
close all;

% Încărcare set de date Iris
IrisData = readtable('iris.data', 'FileType', 'text', 'Delimiter', ',', 'ReadVariableNames', false);

% Extragere caracteristici și etichete
caracteristici = IrisData{:, 1:end-1}; % Extragerea tuturor coloanelor numerice
etichete = IrisData{:, end}; % Extragerea ultimei coloane, care conține etichetele

% Împărțirea setului de date în setul de antrenare și cel de testare (80% antrenare, 20% testare)
cv = cvpartition(size(caracteristici, 1), 'HoldOut', 0.2);
test_indices = cv.test;

% Setul de antrenare
X_antr = caracteristici(~test_indices, :);
y_antr = double(strcmp(etichete(~test_indices), 'Iris-setosa')); % Convertirea etichetelor categorice în binare
% testare
X_testare = caracteristici(test_indices, :);
y_testare = double(strcmp(etichete(test_indices), 'Iris-setosa')); % Convertirea etichetelor categorice în binare

% Normalizarea datelor de antrenare și testare
% Calculez media și deviația standard a caracteristicilor pentru normalizare
[X_antr, medie_caracteristici, sigma] = zscore(X_antr);

% Aplic normalizarea pe setul de testare folosind media și deviația standard calculate pentru setul de antrenare
X_testare = (X_testare - medie_caracteristici) ./ sigma;

% Inițializarea parametrilor rețelei
% Numărul de neuroni în stratul ascuns
n = 12; 

% Numărul de caracteristici
[m, ~] = size(X_antr);

% Greutățile pentru stratul de ieșire
x = randn(n, 1) * 0.01;

% Datele de intrare augmentate
X_antrenare_aug = [ones(size(X_antr, 1), 1), X_antr]; 

% Greutățile incluzând termenul de deplasare
W = randn(size(X_antrenare_aug, 2), n) * 0.01; 

% Funcția de activare (GCU)
g = @(z) z .* cos(z);

% Derivata funcției de activare (GCU)
g_prime = @(z) cos(z) - z .* sin(z);

% Funcția de pierdere entropică încrucișată binară
pierdere_entropica = @(y, y_hat) -mean(y .* log(y_hat) + (1 - y) .* log(1 - y_hat));

% Parametri pentru metoda gradientului
iteratii = 20;

% Stocarea evoluției pierderii și timpului
istoric_pierdere_gradient = zeros(1, iteratii);
istoric_norma_gradient = zeros(1, iteratii);
istoric_timp_gradient = zeros(1, iteratii);

for iteratie = 1:iteratii
    % Pornire cronometru
    timp_inceput = tic; 

    % Propagarea înainte:
   % - Se calculează produsul dintre datele de intrare (X_antrenare_aug) și greutățile W, rezultând Z_ascuns.
    Z_ascuns = X_antrenare_aug * W;

   % - Se aplică funcția de activare g(z) pe Z_ascuns pentru a obține A_ascuns.
    A_ascuns = g(Z_ascuns);

   % - Se calculează produsul dintre A_ascuns și greutățile x, iar rezultatul este trecut prin funcția de activare g(z) pentru a obține Y_prev.
    Y_prev = g(A_ascuns * x);

    % Calculul pierderii
    pierdere = pierdere_entropica(y_antr, Y_prev);
    istoric_pierdere_gradient(iteratie) = pierdere;

    % Calculul gradientului
    gradient_x = (A_ascuns' * (Y_prev - y_antr)) / m;

   % Calculul normei gradientului
    % Frobenius norm
    norma_gradient = norm(gradient_x, 'fro'); 
    istoric_norma_gradient(iteratie) = norma_gradient;

    % Actualizarea parametrilor
    x = x - gradient_x;

    % Stocarea timpului trecut
    timp_trecut = toc(timp_inceput);
    istoric_timp_gradient(iteratie) = timp_trecut;

    % Afișarea pierderii și normei gradientului 
    fprintf('Iterație %d, Pierdere: %.4f, Norma Gradientului: %.4f\n', iteratie, pierdere, norma_gradient);
end
% Calcul total al timpului
t = cumsum(istoric_timp_gradient);

% Plotare evoluție pierdere și norma gradientului în funcție de iteratie
figure;
subplot(2,2,1);
plot(1:iteratii, istoric_pierdere_gradient);
title('Evoluția pierderii');
xlabel('Iteratie');
ylabel('Pierdere');
grid on;

subplot(2,2,2);
plot(1:iteratii, istoric_norma_gradient);
title('Evoluția normei gradientului');
xlabel('Iteratie');
ylabel('Norma Gradientului');
grid on;

% Plotare evoluție pierdere și norma gradientului în funcție de timp
subplot(2,2,3);
plot(t, istoric_pierdere_gradient);
title('Evoluția pierderii în funcție de timp');
xlabel('Timp (secunde)');
ylabel('Pierdere');
grid on;

subplot(2,2,4);
plot(t, istoric_norma_gradient);
title('Evoluția normei gradientului în funcție de timp');
xlabel('Timp (secunde)');
ylabel('Norma Gradientului');
grid on;

% Propagarea înainte pentru setul de testare
Z_ascuns_test = [ones(size(X_testare, 1), 1), X_testare] * W;
A_ascuns_test = g(Z_ascuns_test);
Y_pred_test = g(A_ascuns_test * x);

% Calculul acurateței pe setul de testare
predictii = Y_pred_test > 0.5; % Aplicarea pragului de 0.5 pentru a obține predicții binare
predictii = double(predictii);
acuratete = mean(predictii == y_testare); % Procentul de corectitudine
% Afișarea acurateței pe setul de testare
fprintf('Acuratețea pe setul de testare este: %.2f%%\n', acuratete * 100);

% Crearea matricei de confuzie
matrice_confuzie = confusionmat(y_testare, predictii);

% Afișarea matricei de confuzie
disp(matrice_confuzie);
