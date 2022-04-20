function [pattern_cell, word_cell] = loadSpecialKeywords()
    path = ".\docs\special_wfv\";
    files = dir(path);
    files = files(3:end);
    pattern_cell = {};
    word_cell = {};
    iterator = 1;
    for file = files
        pattern = load(path+file.name);
        word_cell{iterator} = file.name(1:end-4);
        pattern_cell{iterator} = pattern;
    end
end