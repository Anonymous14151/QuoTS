%Main functions to extract scores from subsequences for a specific query


function score = QUEST_sort(X, win_size, query)
    key_vector1 = extract_keywords(X, win_size, "1");
    key_vector2 = extract_keywords(X, win_size/2, "0");
    key_vector3 = extract_keywords(X, win_size/8, "0");
    
    key_vectors = {key_vector1, key_vector2, key_vector3};
    score = zeros(length(query), size(X, 1));
    for q_i = 1:length(query)
        score(q_i, :) = calcualte_query_score(X, query{q_i}, key_vectors);
    end
end

function [moving_noise, moving_clean, moving_up, moving_down, moving_flat, moving_symmetry, moving_complexity, moving_simple, moving_high, moving_low, moving_top, moving_bottom] = moving_feature_extraction(sig, win_size)
%Moving window keywords extraction method. It returns a signal for
%each keyword with the same size of the original signal, because it
%mirrors the original signal with half the size of the defined
%window. This process also prevents border issues.

    half_win_size = fix(win_size/2);
    mirrored_sig = [sig(1)*ones(1,half_win_size), sig, sig(end)*ones(1,half_win_size)];
    
    moving_noise = zeros(1, length(sig));
    moving_up = zeros(1, length(sig));
    moving_down = zeros(1, length(sig));
    moving_flat = zeros(1, length(sig));
    moving_symmetry = zeros(1, length(sig));
    moving_complexity = zeros(1, length(sig));
    moving_high = zeros(1, length(sig));
    moving_top = zeros(1, length(sig));
    
    
    for i = half_win_size:length(sig)+half_win_size-1
        sec_mirrored_sig = mirrored_sig(i-half_win_size+1:i+half_win_size);
        %get signal windowed

        %extract features
        %Noise-----------------------------------------------------------------------------
        moving_noise(i-half_win_size+1) = cum_wbr(sec_mirrored_sig, win_size);
        %Complexity------------------------------------------------------------------------
        moving_complexity(i-half_win_size+1) = complexity_estimation(sec_mirrored_sig);
        %Symmetry--------------------------------------------------------------------------
        moving_symmetry(i-half_win_size+1) = symmetry_estimation(sec_mirrored_sig);
        %Up, Down and Flat-----------------------------------------------------------------
        [up, down] = slope_estimation(sec_mirrored_sig);
        moving_up(i-half_win_size+1) = up;
        moving_down(i-half_win_size+1) = down;        
        moving_flat(i-half_win_size+1) = sum(abs(sec_mirrored_sig-mean(sec_mirrored_sig)));
        moving_high(i-half_win_size+1) = max(sec_mirrored_sig)-min(sec_mirrored_sig);
        moving_top(i-half_win_size+1) = mean(sec_mirrored_sig);
    end            
    
    moving_flat = max(moving_flat) - moving_flat;
    moving_clean = max(moving_noise) - moving_noise;
    moving_simple = max(moving_complexity) - moving_complexity;
    moving_low = max(moving_high) - moving_high;
    moving_bottom = max(moving_top) - moving_top; 
end

function [valley, peak, uval, vval] = extract_mass_features(sig, win_size)
    valley = gauss_val_moving(sig, 0.3, win_size);
    peak = gauss_peak_moving(sig, 0.3, win_size);
    uval = u_gauss_val_moving(sig, win_size);
    vval = v_gauss_val_moving(sig, win_size);
end

function [query_peak, mass_peak] = gauss_peak_moving(sig, c, win_size)
    half_win_size = fix(win_size/2);
    sig_ = [sig(1)*ones(1,half_win_size), sig, sig(end)*ones(1,half_win_size)];
    a = 1;
    b = 0;
    x = linspace(-1, 1, win_size);
    query_peak = a*exp(((x-b).^2)/(-2*(c^2)));
    mass_peak = log10(MASS_V2(sig_, query_peak));
    mass_peak = max(mass_peak) -mass_peak(1:length(sig));
end
        
function [valley_, mass_valley] = gauss_val_moving(sig, c, win_size)
    half_win_size = fix(win_size/2);
    sig_ = [sig(1)*ones(1,half_win_size), sig, sig(end)*ones(1,half_win_size)];
    a = 1;
    b = 0;
    x = linspace(-1, 1, win_size);
    valley_ = -a*exp(((x-b).^2)/(-2*(c^2)));
    mass_valley = log(MASS_V2(sig_, valley_));
    mass_valley = max(mass_valley) - mass_valley(1:length(sig));
end

function mass_v_valley = v_gauss_val_moving(sig, win_size)
    half_win_size = fix(win_size/2);
    sig_ = [sig(1)*ones(1, half_win_size), sig, sig(end)*ones(1, half_win_size)];
    a = 1;
    b = 0;
    
    x = linspace(-1, 1, win_size);
    valley_ = -a*exp(((x-b).^2)/(-2*(0.2^2)));
    
    mass_v_valley = log(MASS_V2(sig_, valley_));
    mass_v_valley(isinf(mass_v_valley)) = max(mass_v_valley(~isinf(mass_v_valley)));
    mass_v_valley(isnan(mass_v_valley)) = max(mass_v_valley(~isnan(mass_v_valley)));
    mass_v_valley = max(mass_v_valley) - mass_v_valley(1:length(sig));
end

function mass_u_valley = u_gauss_val_moving(sig, win_size)
    half_win_size = fix(win_size/2);
    sig_ = [sig(1)*ones(1, half_win_size), sig, sig(end)*ones(1, half_win_size)];
    a = 1;
    b = 0;
    x = linspace(-1, 1, win_size);
    valley_ = -a*exp(((x-b).^2)/(-2*(0.5^2)));
    mass_u_valley = log(log(MASS_V2(sig_, valley_)));
    mass_u_valley(isinf(mass_u_valley)) = max(mass_u_valley(~isinf(mass_u_valley)));
    mass_u_valley(isnan(mass_u_valley)) = max(mass_u_valley(~isnan(mass_u_valley)));
    mass_u_valley = max(mass_u_valley) - mass_u_valley(1:length(sig));
end

function key_vector = extract_keywords(X, win_size, type_)
    keys = {'noise', 'clean', 'up', 'down', 'flat', 'symmetric', 'complex', 'simple', 'peak', 'valley','high', 'low','top','bottom', 'uval', 'vval'};

    NOISE_ = zeros(size(X,1), size(X,2));
    UP_ = zeros(size(X,1), size(X,2));
    DOWN_ = zeros(size(X,1), size(X,2));
    FLAT_ = zeros(size(X,1), size(X,2));
    SYMMETRIC_ = zeros(size(X,1), size(X,2));
    COMPLEX_ = zeros(size(X,1), size(X,2));
    PEAK_ = zeros(size(X,1), size(X,2));
    VALLEY_ = zeros(size(X,1), size(X,2));
    UVAL_ = zeros(size(X, 1), size(X,2));
    VVAL_ = zeros(size(X, 1), size(X,2));
    HIGH_ = zeros(size(X,1), size(X,2));
    LOW_ = zeros(size(X,1), size(X,2));
    TOP_ = zeros(size(X,1), size(X,2));
    BOTTOM_ = zeros(size(X,1), size(X,2));

    %nbr of signals
    s = size(X, 1);
    if(type_=="1")
        [NOISE_, CLEAN_, UP_, DOWN_, FLAT_, SYMMETRIC_, COMPLEX_, SIMPLE_, HIGH_, LOW_, TOP_, BOTTOM_, PEAK_, VALLEY_, UVAL_, VVAL_] = single_class_est(X);
    else
        for i = 1:s
            [NOISE_(i,:), CLEAN_, UP_(i,:), DOWN_(i,:), FLAT_(i,:), SYMMETRIC_(i,:), COMPLEX_(i,:), SIMPLE_, HIGH_(i,:), LOW_(i,:), TOP_(i,:), BOTTOM_(i,:)] = moving_feature_extraction(X(i,:), win_size);
            [VALLEY_, PEAK_, UVAL_, VVAL_] = extract_mass_features(X(i,:), win_size);
        end
    end
    
    %normalize all
    vals = {norm_1_sig(NOISE_), norm_1_sig(CLEAN_), norm_1_sig(UP_), norm_1_sig(DOWN_), norm_1_sig(FLAT_), norm_1_sig(SYMMETRIC_), norm_1_sig(COMPLEX_), norm_1_sig(SIMPLE_), norm_1_sig(PEAK_), norm_1_sig(VALLEY_), norm_1_sig(HIGH_), norm_1_sig(LOW_), norm_1_sig(TOP_), norm_1_sig(BOTTOM_), norm_1_sig(UVAL_), norm_1_sig(VVAL_)};
    %keyvector
    key_vector = containers.Map(keys, vals);
end

function [noise, clean, up, down, flat, symmetry, complexity, simple, high, low, top, bottom, peak, valley, uval, vval] = single_class_est(X)
    noise = zeros(size(X,1), size(X,2));
    up = zeros(size(X,1), size(X,2));
    down = zeros(size(X,1), size(X,2));
    flat = zeros(size(X,1), size(X,2));
    symmetry = zeros(size(X,1), size(X,2));
    complexity = zeros(size(X,1), size(X,2));
    high = zeros(size(X,1), size(X,2));
    top = zeros(size(X,1), size(X,2));
    peak = zeros(size(X,1), size(X,2));
    valley = zeros(size(X,1), size(X,2));
    uval = zeros(size(X,1), size(X,2));
    vval = zeros(size(X,1), size(X,2));
    
    for i = 1:size(X, 1)
        [up_i, down_i] = slope_estimation(X(i, :));
        noise(i, :) = cum_wbr(X(i,:), size(X,1)/4);
        up(i, :) = up_i*ones(1, size(X,2));

        down(i, :) = down_i*ones(1, size(X,2));
        valley(i, :) = max(gauss_val_moving(X(i,:), 0.3, size(X, 2)))*ones(1, size(X,2));
        peak(i, :) = max(gauss_peak_moving(X(i,:), 0.3, size(X, 2)))*ones(1, size(X,2));
        uval(i, :) = max(u_gauss_val_moving(X(i,:), size(X, 2)))*ones(1, size(X,2));
        vval(i, :) = max(v_gauss_val_moving(X(i,:), size(X, 2)))*ones(1, size(X,2));

        flat(i, :) = sum(abs(X(i,:)-mean(X(i,:))))*ones(1, size(X,2));

        symmetry(i,:) = symmetry_estimation(X(i,:))*ones(1, size(X,2));
        complexity(i,:) = complexity_estimation(X(i,:))*ones(1, size(X,2));
        high(i,:) = (max(X(:,i))-min(X(:,i)))*ones(1, size(X,2));
        top(i,:) = mean(X(i,:))*ones(1, size(X,2));
    end
    
    
    clean = max(max(noise)) - noise;

    flat = max(max(flat)) - norm_1_sig(flat);
    simple = max(max(complexity)) - norm_1_sig(complexity);

    bottom = max(max(top)) - norm_1_sig(top);
    low = max(max(high)) - norm_1_sig(high);
end

function symmetry = symmetry_estimation(sig)
%Search minimum of left matrix profile over the fliped signal.
%The lower the value, the higher is the symmetry.
    query = flip(sig);
    win_size = length(sig);
    half_win_size = fix(win_size/4);
    sig2search = [sig(1)*ones(1, half_win_size), sig, sig(end)*ones(1, half_win_size)];
    mass_distance = MASS_V2(sig2search, query);
    if(isnan(min(mass_distance)))
        symmetry = 0;
    else
        symmetry = real(min(mass_distance));
    end
    symmetry = max(max(symmetry)) - symmetry;
end


function residual_sig = wbr(sig, win_size)
%noise level, based on removing the baseline and getting the residuals
    filt_sig = movmean(sig, win_size);
    residual_sig = sig - filt_sig;
end

function residuals = cum_wbr(sig, win)
%check how noisy is a signal by subtracting the wandering baseline
%to the signal and summing the residuals values (need to check the
%win value)
    res_sig = wbr(sig, win);
    residuals = sum(abs(res_sig));
end

function norm_X = norm_1_sig(X)
    if(max(max(X))==min(min(X)))
        norm_X = zeros(size(X,1), size(X,2));
    else
        norm_X = (X - min(min(X)))/(max(max(X))-min(min(X)));
    end
end

%-------------------------------------------
%           Querying methods
%-------------------------------------------

function scores = calcualte_query_score(X, query, key_vectors)
    %Parse text----------------------------------------------
    %   Idea 2 - Check if keywords with space or brackets
    %   indicating window -> Now using regex!!!
    %--------------------------------------------------------
    %1 - apply regex to query, separating brackets and single
    %keywords
    [bracket_groups, keywords_groups] = regexp(query, "\[.+?\]", "match", "split");
    scores = zeros(size(X, 1), 1);
    %2 - calculate scores for each bracket group and single keyword
    %2.1 - start with brackets
    key_vector_wind = key_vectors{2};
    for bracket_i = bracket_groups
        %convert 2 char
        bracket_content = char(bracket_i);
        %calculate score and add to the previous array
        scores = scores + windowed_query_scores(X, bracket_content(2:end-1), key_vector_wind);
    end
    
    %2.2 - calculate scores for each singular keyword
    merged_groups = strjoin(keywords_groups);
    splitted_keywords = split(merged_groups, " ");
    key_vector_sing = key_vectors{1};
    for i = 1:length(splitted_keywords)
        if(size(char(splitted_keywords(i)),2)==0)
            continue
        else
            keyword_score = single_keyword_score_estimation(splitted_keywords(i), key_vector_sing);
            scores = scores + keyword_score;
        end
    end
end

function norm_mean_scores = windowed_query_scores(X, window_query, key_vector)
    keywords = split(window_query);
    
    %extract keywords for each signal, subwindowed and normalized
    norm_mean_scores = extract_keywords_score(X, keywords, key_vector);
end

function scores = extract_keywords_score(X, keywords, key_vector)
    scores = zeros(size(X, 1), length(keywords));
    len_segments = fix(size(X, 2)/length(keywords));
    for i = 1:length(keywords)
        %check for operators on keyword
        if(contains(keywords(i), '!'))
            keyword = lower(char(keywords(i)));
            keyword = keyword(2:end);
            search_op = 1;
        else
            keyword = lower(char(keywords(i)));
            search_op = 2;
        end

        %will have to be adapted when using sliding window on
        %continuous signal
        if(keyword == '.')
            param_ = zeros(1, size(X, 1));
        else
            data_i = key_vector(keyword);
            param_ = (mean(data_i(:, (i-1)*len_segments+1:i*len_segments), 2));
        end
        
        if(search_op == 1)
            scores(:, i) = 1 - param_;
        elseif(search_op == 2)
            scores(:, i) = param_;
        end
    end
    scores = norm_1_sig(mean(scores, 2));
end

function single_keyword_score = single_keyword_score_estimation(keyword, key_vector)
    %check for operators on keyword
    if(contains(keyword, '!'))
        keyword = lower(char(keyword));
        keyword = keyword(2:end);
        single_keyword_score = 1 - norm_1_vec(mean(key_vector(keyword),2));
    else
        keyword = lower(char(keyword));
        single_keyword_score = norm_1_vec(mean(key_vector(keyword),2));
    end
end

function complexity = complexity_estimation(sig)
%Search how complex a signal is by estimating the absolute distance traveled
%by the signal
    complexity = sqrt(sum(diff(sig).^2));
end

function [up, down] = slope_estimation(sig)
%Estimate the slope of a linear approximation to the signal.
%Basically return m, from y = mx + b
    P = polyfit(linspace(0, length(sig), length(sig)), sig, 1);
    slope_ = P(1);

    if(slope_>0)
        up = slope_;
        down = 0;
    elseif(slope_<0)
        up = 0;
        down = abs(slope_);
    else
        up = 0;
        down = 0;
    end
end

function norm_vec = norm_1_vec(s)
    norm_vec = (s-min(s))/(max(s)-min(s));
end