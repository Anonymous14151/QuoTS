

function plotQuoTS(X, scores, k, win_size)
    %PLOT---------------------------------------------------------
    %from the retrieved scores, display the segments found
    %put scores in same dimension as SIGNAL
    new_scores = zeros(1, length(scores)+win_size);
    new_scores(fix(win_size/2):end-fix(win_size/2)-1) = scores;
    x_scores = linspace(1, length(new_scores), length(new_scores));
    half_win = fix(win_size/2);
    
    %3 - plot score function on axes
    plot(new_scores)

    %4 Order by k-most representative windows
    %5 highlight most important windows
    %6 plot eveything in the corresponding plots
    
    arg_k = zeros(1, k);
    
    subplot(1,4,[1,2,3])
    if(size(X, 1)>1)
        for i = 1:size(X, 1)
            plot(norm_1_sig(X(i, :))+(i-1)*1.5, 'LineWidth', 1, 'Color', '#808080')
            hold on
        end
    else
        plot(norm_1_sig(X), 'LineWidth', 0.8, 'Color', '#808080')
        hold on
    end

    cmap = colormap(jet(100));
    
    for k_i = 1:k
        %find max
        [max_ki, argmax_ki] = max(new_scores);
        %save index to plot around it
        arg_k(k_i) = fix(win_size/2) + argmax_ki;
        
        %remove the score and its surrounding 
        a = argmax_ki-half_win+1;
        b = argmax_ki+half_win;

        new_scores(a:b) = 0;
        %highlight on plot
        if(size(X,1)>1)
            for i = 1:size(X, 1)
                %plot subsequences on signal
                s_ = norm_1_sig(X(i,:));
                subplot(1,4,[1,2,3])
                plot(x_scores(a:b), s_(a:b)+(i-1)*1.5, 'Color', cmap(fix(max_ki*85), :), 'LineWidth', 2)
                %plot ordered on side
                subplot(1,4,4)
                x_i = linspace(-1+i, i-0.5, length(a:b));
                plot(x_i, s_(a:b)+(k-k_i)*max_ki, 'LineWidth', 2, 'Color', cmap(fix(max_ki*85), :))
                hold on
            end            
        else
            s_ = norm_1_sig(X);
            subplot(1,4,[1,2,3])
            plot(x_scores(a:b), s_(a:b), 'Color', 'b', 'LineWidth', 2)
            x_i = linspace(0, 0.5, length(a:b));
            subplot(1,4,4)
            plot(x_i, s_(a:b)+(k-k_i), 'LineWidth', 2)
            hold on
        end
        %keep searching the next higher window
    end
end

function norm_sig = norm_1_sig(s)
    if(max(s)==min(s))
        norm_sig = zeros(1, length(s));
    else
        norm_sig = (s - min(s))/(max(s)-min(s));
    end
end