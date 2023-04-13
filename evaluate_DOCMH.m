function eva = evaluate_DOCMH(XChunk,YChunk,LChunk,XTest,YTest,LTest,param)
    
    eva = cell(param.nchunks,1);
    [nchunks ,~] =  size(XChunk);
    for chunki = 1:nchunks
        fprintf('-----chunk----- %3d\n', chunki);
        
        LTrain = cell2mat(LChunk(1:chunki,:));
        XTrain_new = XChunk{chunki,:};
        YTrain_new = YChunk{chunki,:};
        LTrain_new = LChunk{chunki,:};
        
        %Hash code learning
        if chunki == 1
            tic;
            [Pv,Pt,CC,HH,BB,DD,FF] = train_DOCMH0(XTrain_new,YTrain_new,LTrain_new,param);
            traintime=toc;  % Training Time
            evaluation_info.trainT=traintime;
        else
            tic;
            [Pv,Pt,CC,HH,BB,DD,FF] = train_DOCMH(XTrain_new,YTrain_new,LTrain_new,CC,HH,BB,DD,FF,param);
            traintime=toc;  % Training Time
            evaluation_info.trainT=traintime;
        end

        tic;
        [nt ,~] =  size(XTest);
        e=ones(1,nt);
        BxTest=(Pv*XTest'-DD{1,1}*e)>0;
        ByTest=(Pt*YTest'-DD{1,2}*e)>0;
   
        evaluation_info.compressT=toc;        
        B = BB{1,1};
        BxTrain = B>0;
        ByTrain = BxTrain;
                      
        tic;
        DHamm = hammingDist(BxTest', ByTrain');
        [~, orderH] = sort(DHamm, 2);        
        mapIT = map_rank(LTrain,LTest,orderH'); 
        evaluation_info.Image_VS_Text_MAP50 = mapIT(50);
        evaluation_info.Image_VS_Text_MAP100 = mapIT(100);
        evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
        [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LTrain, LTest);
        evaluation_info.Image_To_Text_Precision = precision_at_k(orderH', LTrain, LTest, param.top_K);

        DHamm = hammingDist(ByTest', BxTrain');
        [~, orderH] = sort(DHamm, 2);        
        mapTI = map_rank(LTrain,LTest,orderH'); 
        evaluation_info.Text_VS_Image_MAP50 = mapTI(50);
        evaluation_info.Text_VS_Image_MAP100 = mapTI(100);
        evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
        [evaluation_info.Text_VS_Image_precision,evaluation_info.Text_VS_Image_recall] = precision_recall(orderH', LTrain, LTest);
        evaluation_info.Text_To_Image_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);

        evaluation_info.testT=toc;
        
        eva{chunki} = evaluation_info;
        clear evaluation_info
        
    end
end
