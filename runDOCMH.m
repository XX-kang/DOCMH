close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./datasets/'));
addpath(genpath('./codes/'));

result_URL = './codes/DOCMH/results_DOCMH/';
if ~isdir(result_URL)
    mkdir(result_URL);
end
output= './codes/DOCMH/results_DOCMH.txt';

%The main file
db = {'MIRFLICKR','IAPRTC-12','NUSWIDE10'};
hashmethod = 'DOCMH';
%loopnbits = [8,32];
loopnbits = [8,16,32,64,128];
param.top_K = 100;
param.maxiter = 5;
iter = [1,2,3,4,5]; 
%iter = [1,2]; 

for dbi = 1 :length(db)
    db_name = db{dbi}; param.db_name = db_name;
    % load dataset  
    eva_info = cell(length(iter),length(loopnbits));
    for iteri = 1: length(iter)
        param.iter = iter(iteri);
        load(['./datasets/',db_name,'.mat']); 
        result_name = [result_URL 'tuning_' db_name '_' hashmethod  '.mat' ];
        if strcmp(db_name, 'IAPRTC-12')
            param.chunksize = 2000;
            clear V_tr V_te
            X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
            R = randperm(size(L,1));
            queryInds = R(1:2000);
            sampleInds = R(2001:end);
            param.nchunks = floor(length(sampleInds)/param.chunksize);
            XChunk = cell(param.nchunks,1);
            YChunk = cell(param.nchunks,1);
            LChunk = cell(param.nchunks,1);
            for subi = 1:param.nchunks-1
                XChunk{subi,1} = X(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
                YChunk{subi,1} = Y(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
                LChunk{subi,1} = L(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
            end
            XChunk{param.nchunks,1} = X(sampleInds(param.chunksize*subi+1:end),:);
            YChunk{param.nchunks,1} = Y(sampleInds(param.chunksize*subi+1:end),:);
            LChunk{param.nchunks,1} = L(sampleInds(param.chunksize*subi+1:end),:);
            XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
            clear X Y L

        elseif strcmp(db_name, 'MIRFLICKR')
            param.chunksize = 2000;
            X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
            R = randperm(size(L,1));
            queryInds = R(1:2000);
            sampleInds = R(2001:end);
            param.nchunks = floor(length(sampleInds)/param.chunksize);

            XChunk = cell(param.nchunks,1);
            YChunk = cell(param.nchunks,1);
            LChunk = cell(param.nchunks,1);
            for subi = 1:param.nchunks-1
                XChunk{subi,1} = X(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
                YChunk{subi,1} = Y(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
                LChunk{subi,1} = L(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
            end
            XChunk{param.nchunks,1} = X(sampleInds(param.chunksize*subi+1:end),:);
            YChunk{param.nchunks,1} = Y(sampleInds(param.chunksize*subi+1:end),:);
            LChunk{param.nchunks,1} = L(sampleInds(param.chunksize*subi+1:end),:);

            XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
            clear X Y L
            X_tuning = cell(2,1); X_tuning{1} = XChunk{1}; X_tuning{2} = XChunk{2};
            Y_tuning = cell(2,1); Y_tuning{1} = YChunk{1}; Y_tuning{2} = YChunk{2};
            L_tuning = cell(2,1); L_tuning{1} = LChunk{1}; L_tuning{2} = LChunk{2};       
            
        elseif strcmp(db_name, 'NUSWIDE10')
            param.chunksize = 10000;
            X = [I_tr; I_te]; Y = [T_tr; T_te]; L = [L_tr; L_te];
            R = randperm(size(L,1));
            queryInds = R(1:2000);
            sampleInds = R(2001:end);
            param.nchunks = floor(length(sampleInds)/param.chunksize);

            XChunk = cell(param.nchunks,1);
            YChunk = cell(param.nchunks,1);
            LChunk = cell(param.nchunks,1);
            for subi = 1:param.nchunks-1
                XChunk{subi,1} = X(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
                YChunk{subi,1} = Y(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
                LChunk{subi,1} = L(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
            end
            XChunk{param.nchunks,1} = X(sampleInds(param.chunksize*subi+1:end),:);
            YChunk{param.nchunks,1} = Y(sampleInds(param.chunksize*subi+1:end),:);
            LChunk{param.nchunks,1} = L(sampleInds(param.chunksize*subi+1:end),:);

            XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
            clear X Y L
        end
        clear I_tr I_te L_tr L_te T_tr T_te

        %% Methods
        for ii =1:length(loopnbits)
            fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii));
            param.nbits = loopnbits(ii);
            fprintf('......%s start...... \n\n', 'DOCMH');   
            param.alpha =10^-4; param.beta =10; param.ita =0.1; param.gamma=10;param.xi1=0.5;param.xi2=0.5;
            eva_info_ = evaluate_DOCMH(XChunk,YChunk,LChunk,XTest,YTest,LTest,param);           
            eva_info{iteri,ii} = eva_info_;
            clear eva_info_
        end 
    end
        
        %% save performance
    for ii = 1:length(loopnbits)
        for jj = 1: length(iter)
            %mAP for Table1 with Bits
            ItoT_MAP(jj,ii) = eva_info{jj,ii}{param.nchunks}.Image_VS_Text_MAP;
            TtoI_MAP(jj,ii) = eva_info{jj,ii}{param.nchunks}.Text_VS_Image_MAP; 
            ItoT_MAP50(jj,ii) = eva_info{jj,ii}{param.nchunks}.Image_VS_Text_MAP50;
            TtoI_MAP50(jj,ii) = eva_info{jj,ii}{param.nchunks}.Text_VS_Image_MAP50;
            ItoT_MAP100(jj,ii) = eva_info{jj,ii}{param.nchunks}.Image_VS_Text_MAP100;
            TtoI_MAP100(jj,ii) = eva_info{jj,ii}{param.nchunks}.Text_VS_Image_MAP100;
        end
    end
    average_mAP_ItoT = sum(ItoT_MAP)/length(iter);
    average_mAP_TtoI = sum(TtoI_MAP)/length(iter);
    average_mAP50_ItoT = sum(ItoT_MAP50)/length(iter);
    average_mAP50_TtoI = sum(TtoI_MAP50)/length(iter);
    average_mAP100_ItoT = sum(ItoT_MAP100)/length(iter);
    average_mAP100_TtoI = sum(TtoI_MAP100)/length(iter);
    
    for jj = 1: length(iter)
            %Precision VS Recall
            Image_VS_Text_recall{jj}  = eva_info{jj,3}{param.nchunks}.Image_VS_Text_recall';
            Image_VS_Text_precision{jj} = eva_info{jj,3}{param.nchunks}.Image_VS_Text_precision';
            Text_VS_Image_recall{jj}    = eva_info{jj,3}{param.nchunks}.Text_VS_Image_recall';
            Text_VS_Image_precision{jj} = eva_info{jj,3}{param.nchunks}.Text_VS_Image_precision';  
            %for Fig chunks;           
            for kk = 1:param.nchunks
                % MAP with chunks @32bit
                ave_ItoT_MAP_chunks(jj,kk) = eva_info{jj,3}{kk}.Image_VS_Text_MAP;
                ave_TtoI_MAP_chunks(jj,kk) = eva_info{jj,3}{kk}.Text_VS_Image_MAP;
                ave_ItoT_MAP50_chunks(jj,kk) = eva_info{jj,3}{kk}.Image_VS_Text_MAP50;
                ave_TtoI_MAP50_chunks(jj,kk) = eva_info{jj,3}{kk}.Text_VS_Image_MAP50;
                ave_ItoT_MAP100_chunks(jj,kk) = eva_info{jj,3}{kk}.Image_VS_Text_MAP100;
                ave_TtoI_MAP100_chunks(jj,kk) = eva_info{jj,3}{kk}.Text_VS_Image_MAP100;

                % Top number Precision with chunks
                ave_ItoT_Precision_chunks(jj,kk) = eva_info{jj,3}{kk}.Image_To_Text_Precision(param.top_K);
                ave_TtoI_Precision_chunks(jj,kk) = eva_info{jj,3}{kk}.Text_To_Image_Precision(param.top_K);

                % Training time with chunks
                ave_trainT(jj,kk) = eva_info{jj,3}{kk}.trainT;
            end            
    end     
    average_ItoT_MAP_chunks = sum(ave_ItoT_MAP_chunks)/length(iter);
    average_TtoI_MAP_chunks = sum(ave_TtoI_MAP_chunks)/length(iter);
    average_ItoT_MAP50_chunks = sum(ave_ItoT_MAP50_chunks)/length(iter);
    average_TtoI_MAP50_chunks = sum(ave_TtoI_MAP50_chunks)/length(iter);
    average_ItoT_MAP100_chunks = sum(ave_ItoT_MAP100_chunks)/length(iter);
    average_TtoI_MAP100_chunks = sum(ave_TtoI_MAP100_chunks)/length(iter);
    average_ItoT_Precision_chunks = sum(ave_ItoT_Precision_chunks)/length(iter);
    average_TtoI_Precision_chunks = sum(ave_TtoI_Precision_chunks)/length(iter);
    average_trainT_chunks = sum(ave_trainT)/length(iter);
    save(result_name,'eva_info','average_mAP_ItoT','average_mAP_TtoI',...
        'average_ItoT_MAP_chunks','average_TtoI_MAP_chunks',...
        'average_mAP50_ItoT','average_mAP50_TtoI',...
        'average_ItoT_MAP50_chunks','average_TtoI_MAP50_chunks',...
        'average_mAP100_ItoT','average_mAP100_TtoI',...
        'average_ItoT_MAP100_chunks','average_TtoI_MAP100_chunks',...
        'average_ItoT_Precision_chunks','average_TtoI_Precision_chunks',...
        'Image_VS_Text_recall','Image_VS_Text_precision','Text_VS_Image_recall','Text_VS_Image_precision',...
        'average_trainT_chunks','-v7.3');
    file=fopen(output,'a');
    fprintf(file,'%10s',db_name);
    fprintf(file,'%7.4f',average_mAP_ItoT);
    fprintf(file,'%7.4f',average_mAP_TtoI);
    fprintf(file,'%7.2f\n','');
end

