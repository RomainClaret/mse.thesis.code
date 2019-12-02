# -*- coding: utf-8 -*-
import sys
import time
import networkx as nx
import matplotlib.pyplot as plt

start_time = time.time()
import convex as cx
elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
print("Convex library loaded in",str(elapsed_time))

# python 2 or 3 compatibility selector
if sys.version_info[0] < 3:
    input=raw_input


# Building labels for graph
def get_labels_from_graph(graph):
    labels = dict()
    node_names = [n for n in graph.nodes()]
    for i, node in enumerate(node_names):
        position = node.find("-")
        if position >= 0:
            node_src = node
            node = node[:position]
            node_names[i] = node[:position]
            labels[node_src] = str(cx.wd.wikidata_id_to_label(node))
        else:
            labels[node] = str(cx.wd.wikidata_id_to_label(node))
    return labels


if __name__ == '__main__':
    #initial_question = input("Greetings Human, ask me something: ")
    initial_question = "Which actor voiced the Unicorn in The Last Unicorn?"
    #initial_question = "What is the name of actor who played the character Neo in the movie matrix?"
    print("Auto asking: ",str(initial_question))

    start_time = time.time()
    #print("tagging...")
    result 	= cx.answer_complete_question(initial_question, cx.tagmeToken)
    print(result['answers'])
    print("I think it is:",str(cx.wd.wikidata_id_to_label(result['answers'][0]['answer'])))

    print("creating subgraph..")
    graph 	= cx.gp.expand_context_with_statements(None, [result['context']], qa=True) 
    
    print("Saving Image...")

    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title('Graph - Shapes', fontsize=10)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_size=1000, node_color='yellow', font_size=12, font_weight='bold', with_labels=True, labels=get_labels_from_graph(graph))
    plt.tight_layout()
    plt.savefig("Graph.png", format="PNG")

    #exit("asdasdasd")
    turn = 2
    #next_question = input("Do you want to know something else about it? ")
    next_question = "And Alan Arkin was behind...?"
    print("Auto asking: ",str(next_question))
    answer, graph = cx.answer_follow_up_question(next_question, turn, graph, cx.hyperparameters, cx.number_of_frontier_nodes)
    print(answer)
    sorted_answer = sorted(answer, key = lambda a: a['answer_score'], reverse=True)
    print("I think it is:", str(cx.wd.wikidata_id_to_label(sorted_answer[0]['answer'])))
    #print(cx.number_of_frontier_nodes)

    print("Saving Image...")
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title('Graph - Shapes', fontsize=10)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_size=1500, node_color='yellow', font_size=8, font_weight='bold', labels=get_labels_from_graph(graph))
    plt.tight_layout()
    plt.savefig("Graph2.png", format="PNG")

    print("next step..")



#if cx.telegram_active: cx.telegram.send_message("Greatings Human", cx.telegram_chat_id)

