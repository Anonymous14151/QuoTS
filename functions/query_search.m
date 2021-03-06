%TODO:need to add method for redoing a query based on the input of the
%user...
% 1: parses all words individually, without any operator in it
% 2: checks which is the closest word in the vocabulary
% 3: changes that on the original query, generating a meta-query that is
% understandable by the system
%TODO: make app function callable by script
%try an example on a livescript to put on matlab

%attributes

function scores = query_search(X, K1, K2, K3, query, win_size)
    %Parse text----------------------------------------------
    %   Idea 2 - Check if keywords with space or brackets
    %   indicating window -> Now using regex!!!
    %   Idea 3 (Now in version 4 of the app) - include followed
    %   by/next to/after/before (for now...only using 1 time and
    %   not multiple instances of that operator)
    %--------------------------------------------------------
    
    % 1 - check if multi_part search or single part search:
    search_levels = size(query, 1);
    
    if(search_levels>1)
        %do multilevel search
        [scores, signal_groups] = multi_level_search(query);
    else
        %do singlelevel search
        [scores, signal_groups] = single_level_search(X, K1, K2, K3, query, win_size);
    end
end

function [scores, signal_groups] = single_level_search(X, K1, K2, K3, query, win_size)
    % 2 - check if single, or multiple signal
    [sig_indexs, signal_groups] = regexp(query, "s\d:","match", "split");
    
    % 3 - apply regex to query, separating brackets and single, now
    %also taking care of multisignals
    
    %If there are multiple signals being queried
    if(length(signal_groups)>1)
        if(contains(signal_groups(2), "followed by"))
            %If there is a "followed by" statement for multiple
            %signals, a different approach should be made
            group1 = split(signal_groups(2), "followed by");
            group2 = signal_groups(3);
            scores = norm_1_sig(multi_query_search_followedby(X, K1, K2, K3, [group1(1), group2], sig_indexs, win_size));
        %If normal interaction between signals, than it is the
        %normal addition of scores
        else
            scores = multi_query_search(X, K1, K2, K3, signal_groups, sig_indexs, win_size);
        end
    else
        % 1 - Check the presence of sequence operator
        [followed_by_match, remaining_individual_queries] = regexp(query, "\w+ followed by \w+","match", "split");
        %check scores for followed by
        if(size(followed_by_match, 1)>0)
            score_fb = norm_1_sig(followed_by_search(X, K1, K2, K3, followed_by_match, win_size));
            query = strjoin(remaining_individual_queries);
        else
            query = signal_groups;
        end
        
        scores = uni_query_search(X, K1, K2, K3, char(query), win_size);
        scores = scores + score_fb;
    end

    scores = norm_1_sig(scores);
end

function [new_scores, signal_groups] = recursive_search(query)
    %level 1
    [scores, signal_groups] = single_level_search(query(1, :));

    %level 2
    [scores2, ~] = single_level_search(query(2, :));
    
    %find max
    win_size = WinSize_EditField.Value;
    half_win = fix(win_size/2);
    
    k = K_events_EditField.Value;

    arg_k = zeros(1, k);
    new_scores = zeros(1, length(scores));
    
    for k_i = 1:k
        [~, argmax_ki] = max(scores);
        %save index to plot around it
        
        arg_k(k_i) = fix(win_size/2) + argmax_ki;
        %remove the score and its surrounding 
        a = argmax_ki-half_win+1;
        b = argmax_ki+half_win;
        
        new_scores(a:b) = scores2(a:b);

        scores(a:b) = 0;
    end

    new_scores = norm_1_sig(new_scores);
end

function scores_sig_i = multi_query_search(X, K1, K2, K3, query_groups, signal_indxs, win_size)
    scores_sig_i = zeros(1, size(X, 2)-win_size);
    for i = 1:length(query_groups)-1
        %get the index of the signal you will search the query in
        signal_i = str2num(char(regexp(char(signal_indxs(i)), "\d", 'match')));
        %search the query on the previous signal's index
        scores_sig_i = scores_sig_i + norm_1_sig(multi_uni_query_search(X, K1, K2, K3, char(query_groups(i+1)), signal_i, win_size));
    end
end

function scores = multi_query_search_followedby(X, K1, K2, K3, query_groups, signal_indxs, win_size)
    scores = zeros(1, size(X, 2)-win_size);
    
    signal1 = str2num(char(regexp(char(signal_indxs(1)), "\d", 'match')));
    score_pre = norm_1_sig(multi_uni_query_search(X, K1, K2, K3, char(query_groups(1)), signal1, win_size));

    signal2 = str2num(char(regexp(char(signal_indxs(2)), "\d", 'match')));
    score_pos = norm_1_sig(multi_uni_query_search(X, K1, K2, K3, char(query_groups(2)), signal2, win_size));
    
    scores(1:length(score_pre)-win_size) = score_pre(1:length(score_pre)-win_size) + score_pos(win_size+1:end);
end

function scores_sig_i = multi_uni_query_search(X, K1, K2, K3, query, index_i, win_size)
    %2 - calculate scores for each grouped followed by and single keyword
    %2.1 - start with brackets
    %keywords
    [bracket_groups, keywords_groups] = regexp(query, "\[.+?\]", "match", "split");
    
    scores = zeros(1, size(X, 2)-win_size);
    
    for bracket_i = bracket_groups
        %convert 2 char
        bracket_content = char(bracket_i);
        %calculate score and add to the previous array
        scores = scores + grouped_followed_by_multi(X, K2, K3, bracket_content(2:end-1), index_i, win_size);
    end
    
    %2.2 - calculate scores for each singular keyword
    merged_groups = strjoin(keywords_groups);
    splitted_keywords = split(merged_groups, " ");
    for i = 1:length(splitted_keywords)
        if(size(char(splitted_keywords(i)),2)==0)
            continue
        else
            keyword_score = single_keyword_score_estimation_multi(K1, splitted_keywords(i), index_i, win_size);
            scores = scores + keyword_score;
        end
    end
    
    %the score function for that specific signal (index_i)
    scores_sig_i = scores;
end

function norm_mean_scores = grouped_followed_by_multi(X, K2, K3, window_query, index_i, win_size)
    keywords = split(window_query);
    
    %extract keywords for each signal, subwindowed and normalized
    norm_mean_scores = grouped_followed_by_extract_score_multi(X, K2, K3, keywords, index_i, win_size);
end

function brackets_scores = grouped_followed_by_extract_score_multi(X, KEY_VECTOR2, KEY_VECTOR3, keywords, index_i, win_size)
    if(length(keywords)<3)
        key_vector = KEY_VECTOR2;
    elseif length(keywords)>2
        key_vector = KEY_VECTOR3;
    end
    
    win_sub_size = fix(win_size/length(keywords));
    
    scores = zeros(length(keywords), size(X, 2)-win_size);
    
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

        a = (i-1)*win_sub_size+1;
        b = (size(X, 2)-win_size)+((i-1)*win_sub_size);
        
        %get vector of values associated with the keyword written
        %for a specific window
        %Added a movmean to each keyword when used, based on the
        %win_sub_size.
        if(keyword==".")
            param_vec = zeros(1, b-a+1);
        else
            data_i = key_vector(keyword);
            data_i = movmean(data_i(index_i, :), fix(win_sub_size/2));
            param_vec = data_i(a:b);
        end
        
        if(search_op == 1)
            scores(i, :) = 1 - param_vec;
        elseif(search_op == 2)
            scores(i, :) = param_vec;
        end

    end
    brackets_scores = norm_1_sig(sum(scores, 1));
end

function single_keyword_score = single_keyword_score_estimation_multi(KEY_VECTOR1, keyword, index_i, win_size)
    %check for operators on keyword
    if(contains(keyword, '!'))
        keyword = lower(char(keyword));
        keyword = keyword(2:end);
        key_vector = KEY_VECTOR1(keyword);
        single_keyword_score = 1 - key_vector(index_i, :);
    else
        keyword = lower(char(keyword));
        key_vector = KEY_VECTOR1(keyword);
        single_keyword_score = key_vector(index_i, :);
    end

    a = fix(win_size/2);
    single_keyword_score = single_keyword_score(a:end-a-1);
end

%------------------------------------------------------------------
% Query Score methods
%------------------------------------------------------------------

function scores = grouped_followed_by_extract_score(X, KEY_VECTOR2, KEY_VECTOR3, keywords, win_size)
    
    if(length(keywords)<3)
        key_vector = KEY_VECTOR2;
    elseif length(keywords)>2
        key_vector = KEY_VECTOR3;
    end
    
    win_sub_size = fix(win_size/(length(keywords)));
    
    scores = zeros(length(keywords), length(X)-win_size);
    
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

        a = i*win_sub_size;
        b = (length(X)-win_size)+(i*win_sub_size) - 1;
        
        %Check which keyword is written
        if(keyword==".")
            param_vec = zeros(1, b-a+1);
        else
            data_i = key_vector(keyword);
            param_vec = movmean(data_i(a:b), fix(win_sub_size/2));
        end
        
        if(search_op == 1)
            scores(i, :) = 1 - param_vec;
        elseif(search_op == 2)
            scores(i, :) = param_vec;
        end

    end
    scores = norm_1_sig(sum(scores, 1));
end

function norm_mean_scores = grouped_followed_by(X, K2, K3, window_query, win_size)
    keywords = split(window_query);
    
    %extract keywords for each signal, subwindowed and normalized
    norm_mean_scores = grouped_followed_by_extract_score(X, K2, K3, keywords, win_size);
end

function single_keyword_score = single_keyword_score_estimation(KEY_VECTOR1, keyword, win_size)
    %check for operators on keyword
    if(contains(keyword, '!'))
        keyword = lower(char(keyword));
        keyword = keyword(2:end);
        single_keyword_score = 1 - KEY_VECTOR1(keyword);
    else
        keyword = lower(char(keyword));
        single_keyword_score = KEY_VECTOR1(keyword);
    end
    a = fix(win_size/2);
    single_keyword_score = single_keyword_score(a:end-a-1);
end

function scores = uni_query_search(X, K1, K2, K3, query_group, win_size)
   %keywords
    [bracket_groups, keywords_groups] = regexp(query_group, "\[.+?\]", "match", "split");

    scores = zeros(1, length(X)-win_size);
    %2 - calculate scores for each bracket group and single keyword
    %2.1 - start with brackets
    for bracket_i = bracket_groups
        %convert 2 char
        bracket_content = char(bracket_i);
        %calculate score and add to the previous array
        scores = scores + grouped_followed_by(X, K2, K3, bracket_content(2:end-1));
    end
    
    %2.2 - calculate scores for each singular keyword
    merged_groups = strjoin(keywords_groups);
    splitted_keywords = split(merged_groups, " ");
    for i = 1:length(splitted_keywords)
        if(size(char(splitted_keywords(i)),2)==0)
            continue
        else
            keyword_score = single_keyword_score_estimation(K1, splitted_keywords(i), win_size);
            scores = scores + keyword_score;
        end
    end
end

function scores = followed_by_search(X, K1, K2, K3, query, win_size)
    queries = split(query, "followed by");
    query1 = queries(1);
    query2 = queries(2);
    
    score_pre = uni_query_search(X, K1, K2, K3, char(query1), win_size);
    score_pos = uni_query_search(X, K1, K2, K3, char(query2), win_size);
    scores = zeros(1, length(score_pre));
    scores(1:length(score_pre)-win_size) = score_pre(1:length(score_pre)-win_size) + score_pos(win_size+1:end);
end

function norm_sig = norm_1_sig(s)
    if(max(s)==min(s))
        norm_sig = zeros(1, length(s));
    else
        norm_sig = (s - min(s))/(max(s)-min(s));
    end
end