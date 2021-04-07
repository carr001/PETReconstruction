function x_k = mlem_regularized(x_k_1,z_k_1,w_k_1,gamma,subiter,frame,ith_real)
p1 = mfilename('fullpath');
i=findstr(p1,'\');
p1=p1(1:i(end));
%disp(['current path:',p1,'irt\setup']);
run([p1,'irt\setup']);   
% load MRI3D;
% load PET_3d_allfrm_withoutCSF;
% load u
% load scant
disp(['gamma = ',num2str(gamma),' subiter = ',num2str(subiter),' frame = ',num2str(frame)]);
%% paramters
imgsiz = [160 192];
%% system matrix and sinograms 
sysflag = 1;  % 1: using Fessler's IRT; 
              % 0: using your own system matrix G 
disp('--- Generating system matrix G ...')
if sysflag
    % require Fessler's IRT matlab toolbox to generate a system matrix
    ig = image_geom('nx', imgsiz(1), 'ny', imgsiz(2), 'fov', 33.3);

    % field of view
    ig.mask = ig.circ(16, 16) > 0;

    % system matrix G
    prjsiz = [249 210];%249 radial samples; 210 angular samples
    numprj = prod(prjsiz);
    sg = sino_geom('par', 'nb', prjsiz(1), 'na', prjsiz(2), 'dr', 70/prjsiz(1), 'strip_width', 'dr');
    G  = Gtomo2_strip(sg, ig, 'single', 1);
    Gopt.mtype  = 'fessler';
    Gopt.ig     = ig;
    Gopt.imgsiz = imgsiz;
else
    Gfold = ''; % where you store your system matrix G;
    load([Gfold, 'G']);	% load matrix G
    Gopt.mtype  = 'matlab';
    Gopt.imgsiz = imgsiz;
end
Gopt.disp = 0; % no display of iterations
Gopt.savestep = 1;
%% Genarate Sinograms
% load yi
% load ni
% load ri
load u
load yis_ni_ri
yi = squeeze(yis(ith_real,:,:));

%% MLEM
disp('--- Doing MLEM reconstruction ...')

disp(['mean value of x = ',num2str(mean(x_k_1(:)))]);
tic;
[x_k, ~,~,AT1] = eml_kem(yi(:,frame), ni(:,frame), G, Gopt, x_k_1, ri(:,frame), subiter, []); 
toc;
AT1(u==0) = 0;
%% Regularize
if ~isempty(z_k_1)
    assert(gamma~=0);
    disp('Regularizing...')
    rho   =  z_k_1-w_k_1;
    temp = (AT1./gamma+rho);
    x_k = abs(0.5.*(temp)+0.5.*sqrt(temp.^2-4.*x_k.*AT1./gamma));
end

sz = size(x_k);
pers = sum(isnan(x_k))./prod(sz);
disp(['nan percent = ',num2str(pers)]);
x_k(isnan(x_k)) = 0;
x_k(u==0) = 0;
end
