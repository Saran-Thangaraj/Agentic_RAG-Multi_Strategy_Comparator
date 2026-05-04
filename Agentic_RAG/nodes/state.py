 # nodes/state.py
# Shared memory — stores every agent's result
# Every node reads from state and writes back to state
# Nothing is lost between nodes

def create_state(query, header_chunks, embedding,
                 header_embedding, child_embedding,
                 fixed_embedding, fixed_chunks, child_chunks):

    state = {
        "query":              query,       # original user question
        "chunks":             [],          # filled by retrieve_node
        "grade":              "",          # filled by grade_node
        "answer":             "",          # filled by answer_node
        "retry_count":        0,           # starts at zero
        "max_retry":          3,           # one attempt per strategy
        "current_strategy":   "header",    # which store to try first
        "sections":           [],          # filled by section_node
        "sub_questions":      [],          # filled by decompose_node
        "direct_query":       "",          # filled by rewrite_node
        "classifier":         False,       # filled by classifier_node

        # vector stores and chunks (loaded once in build_pipeline)
        "embedding":          embedding,
        "header_chunks":      header_chunks,
        "header_embedding":   header_embedding,
        "child_embedding":    child_embedding,
        "fixed_embedding":    fixed_embedding,
        "fixed_chunks":       fixed_chunks,
        "child_chunks":       child_chunks,
    }

    return state