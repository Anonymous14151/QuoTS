function [KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3] = word_feature_extraction(X, win_size)
    %if signal is uni-dimensional
    if(size(X, 1)==1)
    %otherwise
        [KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3] = uni_dimensional_extraction(X, win_size);
    elseif(size(X,1)>1)
        [KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3] = multi_dimensional_extraction(X, win_size);
    else
        errordlg('Dimensions of signal incorrect','Error');
    end
end

function [KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3] = uni_dimensional_extraction(signal, win_size)
    
    [keywords_keys1, keywords_vals1] = multi_scale_vector_generation(signal, win_size);
    [keywords_keys2, keywords_vals2] = multi_scale_vector_generation(signal, fix(win_size/4));
    %window 2 = win_size/16
    if(win_size/4<2)
        win_size = 32;
    end
    [keywords_keys3, keywords_vals3] = multi_scale_vector_generation(signal, fix(win_size/8));
    
    KEY_VECTOR1 = containers.Map(keywords_keys1, keywords_vals1);
    KEY_VECTOR2 = containers.Map(keywords_keys2, keywords_vals2);
    KEY_VECTOR3 = containers.Map(keywords_keys3, keywords_vals3);
end

function [KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3] = multi_dimensional_extraction(X, win_size)
    %X - matrix with signals organized by rows
    %win_size - size of the standard window where the search will be made

    len_signal = size(X, 2);
    nbr_signals = size(X, 1);
    
    %main words assigned to features
    keywords_keys = {'noise', 'up', 'down', 'flat', 'symmetric', 'complex', 'high', 'low', 'peak', 'valley', 'stepup', 'stepdown', 'plateauup', 'plateaudown','top','bottom', 'stutter', 'eventful', 'vval', 'uval', 'middle'};
    pre_special_key_size = length(keywords_keys);
    [patterns, words] = loadSpecialKeywords();
    for word_i = length(words)
        keywords_keys{end+1} = words{word_i};
    end

    
    %containers where word feature vectors will be stored and assigned to
    %words
    keywords_vals1 = cell(1, length(keywords_keys));
    keywords_vals2 = cell(1, length(keywords_keys));
    keywords_vals3 = cell(1, length(keywords_keys));
    
    d1 = cell(nbr_signals, length(keywords_keys));
    d2 = cell(nbr_signals, length(keywords_keys));
    d3 = cell(nbr_signals, length(keywords_keys));
    
    for i = 1:nbr_signals
        %sig
        s = X(i,:);
        
        %window 1 = win_size
        d1 = multi_d_scale_vector_generation(d1, s, i, win_size);

        %window 2 = win_size/4
        d2 = multi_d_scale_vector_generation(d2, s, i, fix(win_size/2));

        %window 3 = win_size/8
        d3 = multi_d_scale_vector_generation(d3, s, i, fix(win_size/4));

        for pattern_i = length(patterns)
            d1{i, pre_special_key_size+pattern_i} = special_word_estimation(s, patterns{pattern_i}, win_size);
            d2{i, pre_special_key_size+pattern_i} = special_word_estimation(s, patterns{pattern_i}, win_size/2);
            d3{i, pre_special_key_size+pattern_i} = special_word_estimation(s, patterns{pattern_i}, win_size/4);
        end
    end
    
    for i = 1:length(keywords_keys)
        keywords_vals1{i} = cell2mat(d1(:,i));
        keywords_vals2{i} = cell2mat(d2(:,i));
        keywords_vals3{i} = cell2mat(d3(:,i));
    end

    %map keywords and keys into a container, already normalized
    KEY_VECTOR1 = containers.Map(keywords_keys, keywords_vals1);
    KEY_VECTOR2 = containers.Map(keywords_keys, keywords_vals2);
    KEY_VECTOR3 = containers.Map(keywords_keys, keywords_vals3);
end

function [keywords_keys, keywords_vals] = multi_scale_vector_generation(signal, win_size)
    %load special keywords
    [patterns, words] = loadSpecialKeywords();

    %map keywords and keys into a container, already normalized
    keywords_keys = {'noise', 'up', 'down', 'flat', 'symmetric', 'complex', 'high', 'low', 'peak', 'valley', 'stepup', 'stepdown', 'plateauup', 'plateaudown', 'top', 'bottom', 'stutter', 'eventful', 'vval', 'uval', 'middle'};
    pre_special_key_size = length(keywords_keys);
    for word_i = length(words)
        keywords_keys{end+1} = words{word_i};
    end

    [noise_i, up_i, down_i, flat_i, complex_i, high_i, low_i, top_i, bottom_i, speed_i, mean_i] = quick_moving_feature_extraction(signal, win_size);
    [symmetry_i, stutter_i] = slow_moving_feature_extraction(signal, win_size);
    
    [~, peak_i] = gauss_peak_moving(signal, 0.1, 2*win_size);
    [~, valley_i] = gauss_val_moving(signal, 0.1, 2*win_size);
    
    vval_i = v_gauss_val_moving(signal, 2*win_size);
    uval_i = u_gauss_val_moving(signal, 2*win_size);
    
    [pos_step_i, neg_step_i] = moving_step(signal, 2*win_size);
    [pos_plat_i, neg_plat_i] = moving_plateau(signal, 2*win_size);
%     [uncommon_i, common_i] = MP([fliplr(app.SIGNAL(1:win_size/2)), app.SIGNAL, fliplr(app.SIGNAL(end-win_size/2:end))], win_size);

    keywords_vals = {noise_i, up_i, down_i, flat_i, symmetry_i, complex_i, high_i, low_i, peak_i, valley_i, pos_step_i, neg_step_i, pos_plat_i, neg_plat_i,top_i, bottom_i, stutter_i, speed_i, vval_i, uval_i, mean_i};
    
    %add special keyword to vals array
    for pattern_i = length(patterns)
        keywords_vals{end+1} = special_word_estimation(signal, patterns{pattern_i}, win_size);
    end
end

function d = multi_d_scale_vector_generation(d, s, i, win_size)
    %Extracts a dimension of word feature vectors when signal is
    %multidimensional
    [d{i,1}, d{i,2}, d{i,3}, d{i,4}, d{i,6}, d{i, 7}, d{i, 8}, d{i, 15}, d{i, 16}, d{i,18}, d{i, 21}] = quick_moving_feature_extraction(s, win_size);
    [d{i,5}, d{i,17}] = slow_moving_feature_extraction(s, win_size);

    [~, d{i, 9}] = gauss_peak_moving(s, 0.1, 2*win_size);
    [~, d{i, 10}] = gauss_val_moving(s, 0.1, 2*win_size);
    
    [d{i,11}, d{i,12}] = moving_step(s, win_size*2);
    [d{i,13}, d{i,14}] = moving_plateau(s, win_size*2);

    d{i,19} = v_gauss_val_moving(s, 2*win_size);
    d{i,20} = u_gauss_val_moving(s, 2*win_size);
end

%------------------------------------------------------------------
% Optimized Method to retrieve main features
%------------------------------------------------------------------
function [moving_noise, moving_up, moving_down, moving_flat, moving_complexity, moving_high, moving_low, moving_top, moving_bottom, moving_speed, moving_middle] = quick_moving_feature_extraction(sig, win_size)
    
    half_win_size = fix(win_size/2);

    mirrored_sig = [sig(1)*ones(1, half_win_size), sig, sig(end)*ones(1, half_win_size)];
    mirror_sig_mat = buffer(mirrored_sig, win_size, win_size-1, 'nodelay');
    mirror_sig_mat = mirror_sig_mat(:, 1:length(sig));

    moving_noise = norm_1_sig(movstd(sig, win_size));

    moving_complexity = norm_1_sig(sqrt(movsum(diff(sig).^2, win_size)));
    moving_complexity = [moving_complexity, moving_complexity(end)];
    
    moving_high = norm_1_sig(movmax(sig, win_size) - movmin(sig, win_size));
    moving_low = 1-moving_high;
    
    moving_top = norm_1_sig(movmean(sig, win_size));

    moving_middle = norm_1_sig(abs(sig - movmean(sig, win_size)));
    moving_middle = 1 - moving_middle;

    moving_bottom = 1-moving_top;

    moving_speed = norm_1_sig(movsum(abs(diff(sig)), win_size));
    moving_speed = [moving_speed, moving_speed(end)];

    x_mat = [ones(win_size, 1), transpose(linspace(1, win_size, win_size))];
    
    slopes_ = x_mat\mirror_sig_mat;
    slopes_ = slopes_(2, :);

    moving_up = norm_1_sig(max(slopes_, 0));
    moving_down = norm_1_sig(abs(min(slopes_, 0)));

    moving_flat = norm_1_sig(sum(abs(mirror_sig_mat - mean(mirror_sig_mat, 1)), 1));
    moving_flat = 1 - moving_flat;
end

%------------------------------------------------------------------
% Sliding Method to retrieve moving keywords
%------------------------------------------------------------------

function [moving_symmetry, moving_stutter] = slow_moving_feature_extraction(sig, win_size)

%Moving window keywords extraction method. It returns a signal for
%each keyword with the same size of the original signal, because it
%mirrors the original signal with half the size of the defined
%window. This process also prevents border issues.
    
    half_win_size = fix(win_size/2);
    mirrored_sig = [sig(1)*ones(1, win_size), sig, sig(end)*ones(1, win_size+half_win_size)];
    
    moving_symmetry = zeros(1, length(sig));
    moving_stutter = zeros(1, length(sig));
    
%     for i = win_size:length(sig)+win_size-1
%         sec_mirrored_sig = mirrored_sig(i-half_win_size+1:i+half_win_size);
%         sig2search_symmetry = mirrored_sig(i-win_size+1:i+win_size);
%     
%         %Symmetry--------------------------------------------------------------------------
%         moving_symmetry(i-win_size+1) = app.symmetry_estimation(sig2search_symmetry, fliplr(sec_mirrored_sig));
%     
%         %Symmetry
%         %Down----------------------------------------------------------------------------
%         sec_prev_window = mirrored_sig(i-win_size+1:i);
%         sec_current_window = mirrored_sig(i-fix(win_size/4):i+win_size+fix(win_size/4));
%         moving_stutter(i-win_size+1) = app.stutter_estimation(sec_current_window, sec_prev_window);
%     end            

    %invert symmetry to correctly assign value association
    moving_symmetry = 1 - norm_1_sig(moving_symmetry);

    %Is the next window very similar in shape with the current one?
    moving_stutter = 1 - norm_1_sig(moving_stutter);
end

% function [moving_noise, moving_clean, moving_up, moving_down, moving_flat, moving_symmetry, moving_complexity, moving_simple, moving_high, moving_low, moving_top, moving_bottom] = moving_feature_extraction(sig, win_size)
% %Moving window keywords extraction method. It returns a signal for
% %each keyword with the same size of the original signal, because it
% %mirrors the original signal with half the size of the defined
% %window. This process also prevents border issues.
% 
%     half_win_size = fix(win_size/2);
%     mirrored_sig = [sig(1)*ones(1,half_win_size), sig, sig(end)*ones(1,half_win_size)];
%     
%     moving_noise = zeros(1, length(sig));
%     moving_up = zeros(1, length(sig));
%     moving_down = zeros(1, length(sig));
%     moving_flat = zeros(1, length(sig));
%     moving_symmetry = zeros(1, length(sig));
%     moving_complexity = zeros(1, length(sig));
%     moving_high = zeros(1, length(sig));
%     moving_top = zeros(1, length(sig));
%     
%     
%     for i = half_win_size:length(sig)+half_win_size-1
%         sec_mirrored_sig = mirrored_sig(i-half_win_size+1:i+half_win_size);
%         %get signal windowed
% 
%         %extract features
%         %Noise-----------------------------------------------------------------------------
%         moving_noise(i-half_win_size+1) = cum_wbr(sec_mirrored_sig, win_size);
%         %Complexity------------------------------------------------------------------------
%         moving_complexity(i-half_win_size+1) = complexity_estimation(sec_mirrored_sig);
%         %Symmetry--------------------------------------------------------------------------
%         moving_symmetry(i-half_win_size+1) = symmetry_estimation(sec_mirrored_sig);
%         %Up, Down and Flat-----------------------------------------------------------------
%         [up, down] = slope_estimation(sec_mirrored_sig);
%         moving_up(i-half_win_size+1) = up;
%         moving_down(i-half_win_size+1) = down;        
%         moving_flat(i-half_win_size+1) = sum(abs(sec_mirrored_sig-mean(sec_mirrored_sig)));
%         moving_high(i-half_win_size+1) = max(sec_mirrored_sig)-min(sec_mirrored_sig);
%         moving_top(i-half_win_size+1) = mean(sec_mirrored_sig);
%     end            
%     
%     moving_flat = max(moving_flat) - moving_flat;
%     moving_clean = max(moving_noise) - moving_noise;
%     moving_simple = max(moving_complexity) - moving_complexity;
%     moving_low = max(moving_high) - moving_high;
%     moving_bottom = max(moving_top) - moving_top; 
% end

function [query_peak, mass_peak] = gauss_peak_moving(sig, c, win_size)
    half_win_size = fix(win_size/2);
    sig_ = [sig(1)*ones(1,half_win_size), sig, sig(end)*ones(1,half_win_size)];
    a = 1;
    b = 0;
    x = linspace(-1, 1, win_size);
    query_peak = a*exp(((x-b).^2)/(-2*(c^2)));
    mass_peak = log10(MASS_V2(sig_, query_peak));
    mass_peak = 1 - norm_1_sig(mass_peak(1:length(sig)));
end
        
function [valley_, mass_valley] = gauss_val_moving(sig, c, win_size)
    half_win_size = fix(win_size/2);
    sig_ = [sig(1)*ones(1,half_win_size), sig, sig(end)*ones(1,half_win_size)];
    a = 1;
    b = 0;
    x = linspace(-1, 1, win_size);
    valley_ = -a*exp(((x-b).^2)/(-2*(c^2)));
    mass_valley = log(MASS_V2(sig_, valley_));
    mass_valley = 1 - norm_1_sig(mass_valley(1:length(sig)));
end

function mass_v_valley = v_gauss_val_moving(sig, win_size)
    half_win_size = fix(win_size/2);
    sig_ = [sig(1)*ones(1, half_win_size), sig, sig(end)*ones(1, half_win_size)];
    a = 1;
    b = 0;
    
    x = linspace(-1, 1, win_size);
    valley_ = -a*exp(((x-b).^2)/(-2*(0.05^2)));
    
    mass_v_valley = log(MASS_V2(sig_, valley_));
    mass_v_valley(isinf(mass_v_valley)) = max(mass_v_valley(~isinf(mass_v_valley)));
    mass_v_valley(isnan(mass_v_valley)) = max(mass_v_valley(~isnan(mass_v_valley)));
    mass_v_valley = 1 - norm_1_sig(mass_v_valley(1:length(sig)));
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
    mass_u_valley = 1 - norm_1_sig(mass_u_valley(1:length(sig)));
end

function symmetry = symmetry_estimation(~, sig2search, query)
    %Search minimum of left matrix profile over the fliped signal.
    %The lower the value, the higher is the symmetry.
    mass_distance = MASS_V2(sig2search, query);
    if(isnan(min(mass_distance)))
        symmetry = 0;
    else
        symmetry = real(min(mass_distance));
    end
end

function [pos_step, neg_step] = moving_step(sig, win_size)
    half_win_size = fix(win_size/2);
    sig_ = [fliplr(sig(1:half_win_size)), sig, fliplr(sig(end-half_win_size:end))];

    pos_step_query = zeros(1, win_size);
    neg_step_query = zeros(1, win_size);

    pos_step_query(win_size/2:end) = 1;
    neg_step_query(win_size/2:end) = -1;

    pos_step = MASS_V2(sig_, pos_step_query);
    pos_step = 1 - norm_1_sig(pos_step(1:length(sig)));
    neg_step = MASS_V2(sig_, neg_step_query);
    neg_step = 1 - norm_1_sig(neg_step(1:length(sig)));
end

function [pos_plateau, neg_plateau] = moving_plateau(sig, win_size)
    half_win_size = fix(win_size/2);
    sig_ = [fliplr(sig(1:half_win_size)), sig, fliplr(sig(end-half_win_size:end))];

    pos_plat_query = zeros(1, win_size);
    neg_plat_query = zeros(1, win_size);

    pos_plat_query(win_size/4:3*win_size/4) = 1;
    neg_plat_query(win_size/4:3*win_size/4) = -1;

    pos_plateau = MASS_V2(sig_, pos_plat_query);
    pos_plateau = 1 - norm_1_sig(pos_plateau(1:length(sig)));
    neg_plateau = MASS_V2(sig_, neg_plat_query);
    neg_plateau = 1 - norm_1_sig(neg_plateau(1:length(sig)));
end

function [uncommon, common] = MP(app, s, win_size)
    if(win_size<4)
        win_size = 4;
    end
    [uncommon, ~] = mstamp(transpose(s), fix(win_size), [], []);
    uncommon = app.norm_1_sig(transpose(uncommon(1:length(s)-win_size-1)));
    common = 1 - uncommon;
end

function mass_special = special_word_estimation(sig, query, win_size)
    query_ = transpose(resample(query, ceil(win_size), length(query)));
    half_win_size = fix(win_size/2);
    sig_ = [fliplr(sig(1:half_win_size)), sig, fliplr(sig(end-half_win_size:end))];
    mass_special = 1 - norm_1_sig(log(MASS_V2(sig_, query_)));
    mass_special = mass_special(1:length(sig));
end


function norm_sig = norm_1_sig(s)
    if(max(s)==min(s))
        norm_sig = zeros(1, length(s));
    else
        norm_sig = (s - min(s))/(max(s)-min(s));
    end
end