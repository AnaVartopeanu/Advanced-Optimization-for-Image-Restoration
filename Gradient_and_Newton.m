clc;
clear all;

% Parametrii inițiali
lambda = 0.1;
tol = 1e-6;
max_iter = 100;

% Citirea imaginii și conversia în gri
imagineOriginala = imread('Poza-Comestibila-Daisy_1000x1000.png');
imagineGri = double(rgb2gray(imagineOriginala));

[m, n] = size(imagineGri);
H = fspecial('motion', 20, 45); % Matricea de blur (motion blur)
y_blurata = imfilter(imagineGri, H, 'conv', 'same'); % Imaginea neclară
y = y_blurata(:); % Convertirea imaginii neclare la un vector

% Afisarea imaginii originale
figure;
subplot(1, 2, 1);
imshow(uint8(imagineGri));
title('Imagine Originală');

% Afisarea imaginii blurate
subplot(1, 2, 2);
imshow(uint8(y_blurata));
title('Imagine Blurată');

% Definirea funcțiilor de convoluție
aplica_blur = @(x) reshape(imfilter(reshape(x, m, n), H, 'conv', 'same'), [], 1);

% Verificarea dimensiunilor
assert(length(aplica_blur(zeros(m*n, 1))) == length(y), 'Dimensiunile rezultatelor convoluției sunt incompatibile.');

% Definirea funcției obiectiv
f = @(x) norm(double(y_blurata) - reshape(aplica_blur(x), size(y_blurata)))^2;

% Definirea funcției de constrângere
constrangere = @(x) max(min(x, 255), 0); % Asigură că valorile lui x rămân între 0 și 255

% Apelarea funcțiilor și compararea rezultatelor
x0 = zeros(m * n, 1);

% Metoda Gradientului
[x_grad, iter_grad, timp_grad, f_grad] = metoda_gradientului(aplica_blur, H, y, lambda, tol, max_iter, m, n, f, constrangere);

% Metoda Newton
[x_newton, iter_newton, timp_newton, f_newton] = metoda_newton(aplica_blur, H, y, lambda, tol, max_iter, m, n, f, constrangere);

% Afișarea imaginilor rezultate
figure;
subplot(1, 3, 1); 
imshow(uint8(reshape(x_grad, m, n))); 
title('Imagine Deblurată folosind Metoda Gradientului');

subplot(1, 3, 2);
imshow(uint8(reshape(x_newton, m, n))); 
title('Imagine Deblurată folosind Metoda Newton');


% Plotare evoluție pierdere și norma gradientului în funcție de iterație
figure;
subplot(1, 2, 1);
plot(1:numel(f_grad), f_grad, 'r-', 'LineWidth', 2);
title('Metoda Gradientului: Evoluția Pierderii');
xlabel('Iterații');
ylabel('Pierdere');

subplot(1, 2, 2);
plot(1:numel(f_newton), f_newton, 'b-', 'LineWidth', 2);
title('Metoda Newton: Evoluția Pierderii');
xlabel('Iterații');
ylabel('Pierdere');

% Calculul timpului cumulat pentru graficele evoluției în timp
t_grad = cumsum(timp_grad);
t_newton = cumsum(timp_newton);

% Plotare evoluție pierdere și norma gradientului în funcție de timp
figure;
subplot(1, 2, 1);
semilogy(t_grad, f_grad, 'r-', 'LineWidth', 2);
title('Metoda Gradientului: Evoluția Pierderii în Timp');
xlabel('Timp (secunde)');
ylabel('Pierdere');

subplot(1, 2, 2);
semilogy(t_newton, f_newton, 'b-', 'LineWidth', 2);
title('Metoda Newton: Evoluția Pierderii în Timp');
xlabel('Timp (secunde)');
ylabel('Pierdere');

% Calculul acurateței și matricei de confuzie
prag = 0.5; % Pragul pentru clasificare
imagine_predictata = reshape(x_grad, m, n); % Imaginea deblurată obținută cu metoda Gradientului

% Convertim imaginea deblurată în imagine binară folosind pragul definit
imagine_binara = imagine_predictata > prag;

% Acuratețea și matricea de confuzie
true_positive = sum(imagine_binara(:) == 1 & imagineGri(:) == 255);
false_positive = sum(imagine_binara(:) == 1 & imagineGri(:) == 0);
true_negative = sum(imagine_binara(:) == 0 & imagineGri(:) == 0);
false_negative = sum(imagine_binara(:) == 0 & imagineGri(:) == 255);

acuratete = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative);
matrice_confuzie = [true_positive, false_positive; false_negative, true_negative];

% Afișarea rezultatelor
fprintf('Acuratețea pe imaginea deblurată este: %.2f%%\n', acuratete * 100);
disp('Matricea de confuzie:');
disp(matrice_confuzie);

% Compararea cu fmincon
% [x_fmincon, fval_fmincon, exitflag_fmincon, output_fmincon] = optimizare_fmincon(aplica_blur, y, lambda, x0, m, n);
% fprintf('Metoda fmincon: Cost = %.2f\n', fval_fmincon);

% Bonus: Utilizarea CVX
% cvx_begin
%     variable x(m * n)
%     minimize( norm(y - aplica_blur(x), 2)^2 + lambda * norm(x, 1) )
% cvx_end
% 
% % Afișarea rezultatului CVX
% figure;
% imshow(reshape(x, m, n), []); 
% title('Soluția CVX')

% Funcția pentru Metoda Gradientului
function [x, iter, timp, f_istoric] = metoda_gradientului(aplica_blur, H, y, lambda, tol, max_iter, m, n, f, constrangere)
    x = zeros(m * n, 1); % Inițializare x
    alpha = 5e-1; % Pasul de învățare
    iter = 0;
    f_istoric = zeros(max_iter, 1);
    timp = zeros(max_iter, 1); 

    for k = 1:max_iter
        tic;
        grad = 2 * reshape(imfilter(reshape(aplica_blur(x) - y, m, n), flipud(fliplr(H)), 'conv', 'same'), [], 1) + lambda * sign(x);
        x_nou = x - alpha * grad;

        % Aplic funcția de constrângere
        if ~isempty(constrangere)
            x_nou = constrangere(x_nou);
        end

        timp(k) = toc; % Înregistrez timpul la fiecare iterație

        if norm(x_nou - x, 2) < tol
            break;
        end

        x = x_nou;
        iter = iter + 1;
        f_istoric(iter) = f(x);
    end
    timp = timp(1:iter); 
    f_istoric = f_istoric(1:iter);
    disp(['Metoda Gradientului: ' num2str(iter) ' iteratii, ' num2str(sum(timp), '%.2f') ' secunde, Cost final: ' num2str(f(x))]);
end

% Funcția pentru Metoda Newton
function [x, iter, timp, f_istoric] = metoda_newton(aplica_blur, H, y, lambda, tol, max_iter, m, n, f, constrangere)
    x = zeros(m * n, 1); % Inițializare x
    iter = 0;
    f_istoric = zeros(max_iter, 1);
    timp = zeros(max_iter, 1);

    for k = 1:max_iter
        tic;
        grad = 2 * reshape(imfilter(reshape(aplica_blur(x) - y, m, n), flipud(fliplr(H)), 'conv', 'same'), [], 1) + lambda * sign(x);
        
        % Utilizarea conjugate gradient pentru a rezolva sistemul de ecuații liniare
        hess_mult = @(v) 2 * reshape(imfilter(reshape(imfilter(reshape(v, m, n), H, 'conv', 'same'), m, n), flipud(fliplr(H)), 'conv', 'same'), [], 1) + lambda * (v ./ (abs(x) + 1e-5));
        [delta_x, ~] = pcg(hess_mult, -grad, tol, max_iter);

        x_nou = x + delta_x;

        % Aplic funcția de constrângere
        if ~isempty(constrangere)
            x_nou = constrangere(x_nou);
        end

        timp(k) = toc; % Înregistrez timpul la fiecare iterație

        if norm(delta_x, 2) < tol
            break;
        end

        x = x_nou;
        iter = iter + 1;
        f_istoric(iter) = f(x);
    end
    timp = timp(1:iter); % Reduc dimensiunea variabilei timp
    f_istoric = f_istoric(1:iter);
    disp(['Metoda Newton: ' num2str(iter) ' iteratii, ' num2str(sum(timp), '%.2f') ' secunde, Cost final: ' num2str(f(x))]);
end

% Funcție pentru fmincon
% function [x, fval, exitflag, output] = optimizare_fmincon(aplica_blur, y, lambda, x0, m, n)
%     obj_fun = @(x) norm(y - aplica_blur(x), 2)^2 + lambda * norm(x, 1);
%     options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'iter');
%     [x, fval, exitflag, output] = fmincon(obj_fun, x0, [], [], [], [], [], [], [], options);
% end
