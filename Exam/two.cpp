#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

struct Contact {
    string id;
    string name;
    string phone;
};

vector<Contact> string_to_contacts(const string &text) {
    vector<Contact> contacts;
    istringstream iss(text);
    string line;
    while (getline(iss, line)) {
        if (line.empty()) continue;
        int firstDel = line.find("//");
        int secondDel = line.find("//", firstDel + 2);
        if (firstDel == string::npos || secondDel == string::npos) continue;

        string id = line.substr(0, firstDel);
        string name = line.substr(firstDel + 2, secondDel - (firstDel + 2));
        string phone = line.substr(secondDel + 2);

        contacts.push_back({id, name, phone});
    }
    return contacts;
}

string vector_to_string(const vector<Contact> &contacts, int start, int end) {
    string result;
    for (int i = start; i < min((int)contacts.size(), end); i++) {
        result += contacts[i].id + "//" + contacts[i].name + "//" + contacts[i].phone + "\n";
    }
    return result;
}

string contact_to_string(const Contact &c) {
    return c.id + "//" + c.name + "//" + c.phone + "\n";
}

void read_phonebook(const vector<string> &files, vector<Contact> &contacts) {
    for (const string &file : files) {
        ifstream f(file);
        string line;
        while (getline(f, line)) {
            if (line.empty()) continue;
            int firstDel = line.find("//");
            int secondDel = line.find("//", firstDel + 2);
            if (firstDel == string::npos || secondDel == string::npos) continue;

            string id = line.substr(0, firstDel);
            string name = line.substr(firstDel + 2, secondDel - (firstDel + 2));
            string phone = line.substr(secondDel + 2);

            contacts.push_back({id, name, phone});
        }
    }
}

void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

string receive_string(int sender) {
    int len;
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    string res(buf);
    delete[] buf;
    return res;
}

optional<Contact> check(const Contact &c, const string &search) {
    if (c.phone.find(search) != string::npos) {
        return c;
    }
    return nullopt;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -n <procs> " << argv[0] << " <file>... <search_term>\n";
        MPI_Finalize();
        return 1;
    }

    string search_term = argv[argc - 1];
    double start, end;

    if (rank == 0){
        vector<string> files(argv + 1, argv + argc - 1);
        vector<Contact> contacts;
        read_phonebook(files, contacts);

        int total = contacts.size();
        int chunk = (total + size - 1) / size;

        for (int i = 1; i < size; i++) {
            string text = vector_to_string(contacts, i * chunk, (i + 1) * chunk);
            send_string(text, i);
        }

        start = MPI_Wtime();
        vector<Contact> matched_contacts;
        for (int i = 0; i < min(chunk, total); i++) {
            auto match = check(contacts[i], search_term);
            if (match) matched_contacts.push_back(*match);
        }

        for (int i = 1; i < size; i++) {
            string recv = receive_string(i);
            vector<Contact> matches = string_to_contacts(recv);
            matched_contacts.insert(matched_contacts.end(), matches.begin(), matches.end());
        }

        sort(matched_contacts.begin(), matched_contacts.end(), [](const Contact &a, const Contact &b) {
            return a.name < b.name;
        });

        ofstream out("output.txt");
        for (const auto &c : matched_contacts) {
            out << c.id << " " << c.name << " " << c.phone << "\n";
        }
        out.close();

        end = MPI_Wtime();
        printf("Process %d took %f seconds.\n", rank, end - start);

    } else {
        string recv_text = receive_string(0);
        vector<Contact> contacts = string_to_contacts(recv_text);

        start = MPI_Wtime();
        string result;
        for (auto &c : contacts) {
            auto match = check(c, search_term);
            if (match) result += contact_to_string(*match);
        }
        end = MPI_Wtime();

        send_string(result, 0);
        printf("Process %d took %f seconds.\n", rank, end - start);
    }

    MPI_Finalize();
    return 0;
}
