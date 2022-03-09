function scores = QuoTS_search(X, win_size, query)
    %1 - extract word feature vectors
    [KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3] = word_feature_extraction(X, win_size);
    %2 - performs search on the signal
    scores = query_search(X, KEY_VECTOR1, KEY_VECTOR2, KEY_VECTOR3, query, win_size);
end