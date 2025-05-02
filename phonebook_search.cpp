/*
    How to run:
    mpic++ -o search phonebook_search.cpp
    mpirun -np 4 ./search phonebook1.txt Bob
    mpirun -np 4 ./search file1.txt file2.txt Bob //multiple files
*/

#include <bits/stdc++.h>
#include<mpi.h>
using namespace std;

//structure to store name-phone pairs
struct Contact {
    string name;
    string phone;
};

// Parses a string of "name,phone" lines into Contact vector
vector<Contact> string_to_contacts(const string &text) {
    vector<Contact> contacts;
    istringstream iss(text);
    string line;
    while (getline(iss, line)) {
        if (line.empty()) continue;
        int comma = line.find(",");
        if (comma == string::npos) continue;
        contacts.push_back({line.substr(0, comma), line.substr(comma + 1)});
    }
    return contacts;
}

// Receives a string from a sender process using MPI
string receive_string(int sender) {
    int len;
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive length
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive content
    string res(buf);
    delete[] buf;
    return res;
}

//Checks if the contact name contains the search term 
string check(const Contact &c, const string &search){
    if(c.name.find(search) != string::npos){
        return c.name + " " + c.phone + "\n"; //match found
    }
    return "";
}

// Sends a string to a receiver process using MPI
void send_string(const string &text, int receiver) {
    int len = text.size() + 1; // Include null terminator
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);         // Send length
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD); // Send content
}

//Converts a slice of Contact vector into a single string
string vector_to_string(const vector<Contact> &contacts, int start, int end){
    string result;
    for(int i=start; i<min((int)contacts.size(), end); i++){
        result += contacts[i].name + "," + contacts[i].phone + "\n";
    }
    return result;
}

// Reads phonebook file(s) into the contacts vector
void read_phonebook(const vector<string> &files, vector<Contact> &contacts) {
    for (const string &file : files) {
        ifstream f(file); //Open the file for reading.
        string line;
        while (getline(f, line)) {//Read the file line by line.
            if (line.empty()) continue;
            int comma = line.find(",");
            if (comma == string::npos) continue;
            // Remove any extra formatting and extract name and phone
            contacts.push_back({line.substr(1, comma - 2), line.substr(comma + 2, line.size() - comma - 3)});
        }
    }
}

int main(int argc, char **argv){
    //Initialize MPI environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //Get process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size); //Get total number of processes

    // Require at least one file and a search term
    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -n <procs> " << argv[0] << " <file>... <search>\n";
        MPI_Finalize();
        return 1;
    }

    //Last argument is the search sterm
    string search_term = argv[argc-1];
    double start, end;

    if(rank == 0){//root process
        //Rank 0 read all file name from command line
        vector<string> files(argv+1, argv+argc-1);//{phonebook1.txt}
        vector<Contact> contacts;
        read_phonebook(files, contacts); // Load contacts from files

        int total = contacts.size();
        int chunk = (total + size - 1) / size; // Divide work among processes

        //Distribute contact chunks to worker processes
        for(int i=1;i<size;i++){
            string text = vector_to_string(contacts, i*chunk, (i+1)*chunk);
            send_string(text, i);
        }

        //Root process does its own chunk
        start = MPI_Wtime();
        string result;
        for(int i=0;i<min(chunk, total); i++){
            string match = check(contacts[i], search_term);
            if(!match.empty()) result += match;
        }
        end = MPI_Wtime();

        //Collect result from other processes
        for(int i=1;i<size;i++){
            string recv = receive_string(i);
            if(!recv.empty()) result += recv;
        }

        //cout << "\nSearch Results:\n" << result << endl; //to show result on console

        //save result to file
        ofstream out("output.txt");
        out << result;
        out.close();

        // Report time taken
        printf("Process %d took %f seconds.\n", rank, end - start);



    }
    else{
        // Worker process receives chunk from root
        string recv_text = receive_string(0);
        vector<Contact> contacts = string_to_contacts(recv_text);

        // Perform search
        start = MPI_Wtime();
        string result;
        for (auto &c : contacts) {
            string match = check(c, search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime();

        // Send results back to root
        send_string(result, 0);

        // Report time taken
        printf("Process %d took %f seconds.\n", rank, end - start);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;

}